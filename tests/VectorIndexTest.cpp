/*
 * Copyright © 2024 MOQI SINGAPORE PTE. LTD.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3.0 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
 */

#include <atomic>
#include <csignal>
#include <memory>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <thread>
#include <SearchIndex/Common/DenseBitmap.h>
#include <SearchIndex/IndexDataFileIO.h>
#include <SearchIndex/VectorSearch.h>
#include <argparse/argparse.hpp>
#include <boost/ut.hpp>
#include <sys/resource.h>

using namespace Search;

#ifdef SI_USE_FILE_IOSTREAM

#    define IStream std::ifstream
#    define OStream std::ofstream

#    define IndexDataReader(path_prefix) \
        IndexDataFileReader<std::ifstream>( \
            path_prefix, \
            [](const std::string & name, std::ios::openmode mode) \
            { return std::make_shared<std::ifstream>(name, mode); });
#    define IndexDataWriter(path_prefix) \
        IndexDataFileWriter<std::ofstream>( \
            path_prefix, \
            [](const std::string & name, std::ios::openmode mode) \
            { return std::make_shared<std::ofstream>(name, mode); });

#else

#    define IStream AbstractIStream
#    define OStream AbstractOStream

#    define IndexDataReader(path_prefix) \
        IndexDataFileReader<AbstractIStream>( \
            path_prefix, \
            [](const std::string & name, std::ios::openmode mode) \
            { return std::make_shared<FileBasedIStream>(name, mode); });
#    define IndexDataWriter(path_prefix) \
        IndexDataFileWriter<AbstractOStream>( \
            path_prefix, \
            [](const std::string & name, std::ios::openmode mode) \
            { return std::make_shared<FileBasedOStream>(name, mode); });

#endif

#ifndef MYSCALE_MODE
// use SPDLOG_LEVEL=debug to allow for debug level logging
#    include "spdlog/cfg/env.h"
#endif

typedef std::function<void(int)> SignalHandlerType;

// Create a global instance of the function object
SignalHandlerType globalSignalHandler;

// Create a wrapper signal handler function
void signalHandlerWrapper(int signal_num)
{
    // Call the function object
    globalSignalHandler(signal_num);
}

template <typename T>
class DataReader : public IndexSourceDataReader<T>
{
public:
    using DataChunk = DataSet<T>;

    DataReader(
        size_t data_dim_,
        size_t num_train_,
        size_t num_data_,
        size_t empty_data_ = 0,
        size_t id_offset = 0,
        size_t data_dim_num_copy_ = 1,
        size_t seed_ = 1248) :
        id_list(num_data_),
        data_dim(data_dim_),
        num_train(num_train_),
        num_data(num_data_),
        empty_data(empty_data_),
        data_dim_num_copy(data_dim_num_copy_),
        num_read(0),
        init_seed(seed_)
    {
        for (size_t j = 0; j < num_data; j++)
            id_list[j] = j + id_offset;
        SI_THROW_IF_NOT_FMT(
            data_dim % data_dim_num_copy == 0,
            ErrorCode::BAD_ARGUMENTS,
            "data_dim %lu must be divisible by data_dim_num_copy %lu",
            data_dim,
            data_dim_num_copy);
    }

    std::shared_ptr<DataChunk> sampleData(size_t n) override
    {
        size_t num_rows = std::min(n, num_train);
        SI_LOG_INFO("sample {} vectors for training", num_rows);
        return generateData(num_read, num_rows);
    }

    std::shared_ptr<DataChunk> getData(size_t pos, size_t num) const
    {
        return generateData(pos, num);
    }

    size_t numDataRead() const override { return num_read; }

    size_t dataDimension() const override { return data_dim; }

    bool eof() override { return num_data - empty_data == num_read; }

    void seekg(std::streamsize offset, std::ios::seekdir dir) override
    {
        SI_THROW_IF_NOT_MSG(
            offset % this->dataSize() == 0,
            ErrorCode::LOGICAL_ERROR,
            "Only support reading whole rows of data");
        size_t rows_offset = offset / this->dataSize();
        switch (dir)
        {
            // only seekg(0, beg) is currently used, so we can just implement this case
            case std::ios::beg:
                num_read = rows_offset;
                break;
            case std::ios::end:
                num_read = num_data - empty_data + rows_offset;
                break;
            case std::ios::cur:
                num_read += rows_offset;
                break;
            default:
                SI_THROW_FMT(
                    ErrorCode::UNSUPPORTED_PARAMETER,
                    "Unsupported seekdir %d",
                    static_cast<int>(dir));
        }
    }

protected:
    std::shared_ptr<DataChunk>
    generateData(size_t offset, size_t num_rows) const
    {
        auto block_size = data_dim / data_dim_num_copy;
        if (offset * 10 / num_data < (offset + num_rows) * 10 / num_data)
            SI_LOG_INFO(
                "DataReader generating {} vectors, offset {} data_dim {} "
                "init_seed {} block_size {}",
                num_rows,
                offset,
                data_dim,
                init_seed,
                block_size);
        std::shared_ptr<DataChunk> chunk;
        if constexpr (!std::is_same<T, bool>::value)
        {
            T * data = new T[data_dim * num_rows];

            // generate data on the fly
#pragma omp parallel for
            for (size_t i = 0; i < num_rows; i++)
            {
                // reset seed for each data point
                std::mt19937 rng(init_seed + offset + i);
                std::uniform_real_distribution<> distrib;
                // generate the first block
                for (size_t j = 0; j < block_size; j++)
                    data[i * data_dim + j] = distrib(rng);
                // copy the first block to the rest blocks
                for (size_t k = 1; k < data_dim_num_copy; ++k)
                {
                    for (size_t j = 0; j < block_size; j++)
                        data[i * data_dim + k * block_size + j]
                            = data[i * data_dim + j];
                }
            }
            chunk = std::make_shared<DataChunk>(
                data, num_rows, data_dim, [data]() { delete[] data; });
        }
        else
        {
            /// set bool at here
            chunk = std::make_shared<DataChunk>(nullptr, num_rows, data_dim);
            std::uniform_int_distribution<int> uniformDist(0, 0xff);

            auto & data_container = chunk->data_container;
            for (size_t i = 0; i < num_rows; i++)
            {
                std::mt19937 bool_rng(init_seed + offset + i);
                for (size_t d = 0; d < chunk->singleVectorSizeInByte(); d++)
                {
                    data_container[i * chunk->singleVectorSizeInByte() + d]
                        = uniformDist(bool_rng);
                }
            }
        }

        chunk->setDataID(&id_list[offset]);
        // perform normalization in place
        if (!std::is_same<T, bool>::value)
        {
            chunk = chunk->normalize(true, sqrt(data_dim_num_copy));
        }
        return chunk;
    }

    std::shared_ptr<DataChunk> readDataImpl(size_t n) override
    {
        // empty data rows are not returned
        int num_rows = std::min(n, num_data - empty_data - num_read);
        if (num_rows <= 0)
            return nullptr;

        auto chunk = generateData(num_read, num_rows);
        num_read += num_rows;
        return chunk;
    }

private:
    std::vector<idx_t> id_list;
    size_t data_dim;
    size_t num_train;
    size_t num_data;
    size_t empty_data;
    size_t data_dim_num_copy;
    size_t num_read;
    size_t init_seed;
};

typedef VectorIndex<IStream, OStream, DenseBitmap, DataType::FloatVector>
    FloatVectorIndex;

typedef VectorIndex<IStream, OStream, DenseBitmap, DataType::BinaryVector>
    BinaryVectorIndex;

struct VectorTestActions
{
    static const int CREATE = 1;
    static const int BUILD = 1 << 1;
    static const int SAVE = 1 << 2;
    static const int LOAD = 1 << 3;
    static const int SEARCH = 1 << 4;
};

struct VectorIndexTester
{
    inline static std::unordered_map<Metric, float> MetricToExactDist
        = {{Metric::IP, 1.0f},
           {Metric::L2, 0.0},
           {Metric::Cosine, 0.0},
           {Metric::Jaccard, 0.0},
           {Metric::Hamming, 0.0}};

    Metric metric;
    size_t data_dim;
    size_t data_dim_num_copy;
    size_t max_points;
    int num_build_threads;
    int filter_out_mod;
    int filter_keep_min;
    int filter_keep_max;
    int top_k;
    int scann_build_hashed_dataset_by_token;
    std::string scann_children_per_level;
    int scann_l_search;
    float scann_l_search_ratio;
    int scann_num_reorder;
    int scann_dims_per_block;
    float scann_aq_threshold;
    int use_default_params;
    int rand_seed;
    int stress_threads;
    int stress_sec;
    int abort_sec;
    int check_build_canceled_sec;
    bool abort_sec_set{false};

    VectorIndexTester() { }

    void testBinaryVectorIndex(IndexType index_type, int flags)
    {
        try
        {
            testVectorIndexImpl<bool, DataType::BinaryVector>(
                index_type, flags);
        }
        catch (std::exception & e)
        {
            SI_LOG_ERROR("Exception: {}", e.what());
            // if e.what() contains "index is aborted", just exit the program
            if (std::string(e.what()).find("index is aborted")
                != std::string::npos)
            {
                SI_LOG_INFO("Index aborted, exiting now ...");
                exit(0);
            }
        }
        catch (...)
        {
            SI_LOG_ERROR("Unknown exception");
            exit(1);
        }
    }

    void testFloatVectorIndex(IndexType index_type, int flags)
    {
        try
        {
            testVectorIndexImpl<float, DataType::FloatVector>(
                index_type, flags);
        }
        catch (std::exception & e)
        {
            SI_LOG_ERROR("Exception: {}", e.what());
            // if e.what() contains "index is aborted", just exit the program
            if (std::string(e.what()).find("index is aborted")
                != std::string::npos)
            {
                SI_LOG_INFO("Index aborted, exiting now ...");
                exit(0);
            }
        }
        catch (...)
        {
            SI_LOG_ERROR("Unknown exception");
            exit(1);
        }
    }

    /* @example VectorIndexTest.cpp
 * example for multiple vector index type
 */
    template <typename T, DataType DATA_TYPE>
    void testVectorIndexImpl(IndexType index_type, int flags)
    {
        using namespace boost::ut;
        auto index_type_str = enumToString(index_type);
        Parameters index_params;
        Parameters search_params;

        SI_LOG_INFO(
            "#### testVectorIndexImpl, abort_sec {}, abort_sec_set {}",
            abort_sec,
            abort_sec_set);

        index_params.setParam("load_index_version", std::string("1.0.0"));
        if (index_type_str.starts_with("IVF"))
        {
            // IVF specific parameters
            index_params.setParam("ncentroids", int(4 * sqrt(max_points)));
            search_params.setParam("nprobe", 8);
        }
        if (index_type == IndexType::SCANN)
        {

            if (use_default_params)
                SI_LOG_INFO("Using default parameters for vector index");
            else
            {
                index_params.setParam(
                    "build_hashed_dataset_by_token",
                    scann_build_hashed_dataset_by_token);
                if (scann_dims_per_block > 0)
                    index_params.setParam(
                        "quantization_block_dimension", scann_dims_per_block);
                if (scann_aq_threshold >= 0)
                    index_params.setParam("aq_threshold", scann_aq_threshold);
                index_params.setParam(
                    "num_children_per_level", scann_children_per_level);
                uint32_t num_leaf_nodes = 1;
                for (auto c : scann_children_per_level)
                    num_leaf_nodes *= c;

                size_t min_cluster_size
                    = std::max(1UL, std::min(50UL, 5 * max_points / 1000));
                size_t training_sample_size = std::min(
                    max_points, std::max(1000UL, 100UL * num_leaf_nodes));
                index_params.setParam("min_cluster_size", min_cluster_size);
                index_params.setParam(
                    "training_sample_size", training_sample_size);

                if (data_dim >= 16)
                {
                    search_params.setParam("alpha", 3.0f);
                    if (scann_l_search > 0)
                        search_params.setParam("l_search", scann_l_search);
                    if (scann_num_reorder > 0)
                        search_params.setParam("num_reorder", scann_num_reorder);
                }
                else
                {
                    // for small dimensions, search all cluster for better recall
                    search_params.setParam("l_search", 20);
                    search_params.setParam("num_reorder", 1000);
                }
            }
            // override default l_search_ratio
            if (scann_l_search_ratio > 0)
                search_params.setParam("l_search_ratio", scann_l_search_ratio);
        }
        std::shared_ptr<VectorIndex<IStream, OStream, DenseBitmap, DATA_TYPE>>
            index;
        SI_LOG_INFO("{}: index_type {}", __func__, enumToString(index_type));
        size_t num_train = std::min(max_points, 100 * 1000UL);
        size_t empty_data = max_points >= 2000 ? 10 : 0;
        size_t id_offset = 1000;
        size_t seed = rand_seed;

        if (flags & VectorTestActions::CREATE)
        {
            index = createVectorIndex<IStream, OStream, DenseBitmap, DATA_TYPE>(
                "index_" + enumToString(index_type),
                index_type,
                metric,
                data_dim,
                max_points,
                index_params,
                /* file_store_prefix */ "test_vector_index_",
                /* use_file_checksum */ true);
            expect(index.get() != nullptr);
        }

        DataReader<T> data_reader(
            data_dim,
            num_train,
            max_points,
            empty_data,
            id_offset,
            data_dim_num_copy,
            seed);
        auto index_save_prefix
            = "/tmp/test_vector_index_" + index->getName() + "_";

        std::shared_ptr<std::thread> signal_abort_thread;

        sigset_t signal_set;
        sigemptyset(&signal_set);
        sigaddset(&signal_set, SIGINT);
        sigaddset(&signal_set, SIGUSR1);
        auto abort_index = [&]()
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            index->abort();
            auto t1 = std::chrono::high_resolution_clock::now();
            auto time_ms
                = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                      .count();
            SI_LOG_INFO("Index abort took {} ms.", time_ms);
            // abort takes less than 200ms
            boost::ut::expect(time_ms <= 1000);
        };
        auto signal_abort_handler = [&]()
        {
            SI_LOG_INFO("signal handler thread started");
            if (abort_sec == 0)
            {
                int signal_num;
                sigwait(&signal_set, &signal_num);
                if (signal_num == SIGINT)
                {
                    SI_LOG_INFO(
                        "Interrupt signal ({}) received, abortping the index "
                        "now",
                        signal_num);
                    abort_index();
                }
                else
                    SI_LOG_INFO("Receiving other signals, exiting now.");
            }
            else
            {
                SI_LOG_INFO(
                    "Sleep for {} seconds before aborting the index",
                    abort_sec);
                std::this_thread::sleep_for(std::chrono::seconds(abort_sec));
                SI_LOG_INFO("Wake up and abortping the index");
                abort_index();
            }
        };

        if (abort_sec == 0 || (abort_sec > 0 and abort_sec_set))
            signal_abort_thread
                = std::make_shared<std::thread>(signal_abort_handler);
        OnExit abort_thread_join(
            [&]
            {
                if (!signal_abort_thread)
                    return;
                if (abort_sec == 0)
                {
                    // raise SIGUSR1 manually to abort the signal_abort_handler thread
                    SI_LOG_INFO(
                        "Sending SIGUSR1 signal to abort the signal thread");
                    pthread_kill(signal_abort_thread->native_handle(), SIGUSR1);
                }
                signal_abort_thread->join();
            });

        if (flags & VectorTestActions::BUILD)
        {
            SI_LOG_INFO(
                "Building vector index {}, progress {} num_build_threads {}",
                index->getName(),
                index->getBuildProgress(),
                num_build_threads);
            // index building takes a while
            omp_set_num_threads(num_build_threads);
            auto t0 = std::chrono::high_resolution_clock::now();
            std::function<bool()> check_build_canceled = nullptr;
            if (check_build_canceled_sec > 0)
            {
                // cancel build after check_build_canceled_sec seconds
                check_build_canceled = [&t0, this]()
                {
                    auto t = std::chrono::high_resolution_clock::now();
                    auto sec = std::chrono::duration_cast<std::chrono::seconds>(
                                   t - t0)
                                   .count();
                    return sec > check_build_canceled_sec;
                };
            }
            index->build(&data_reader, num_build_threads, check_build_canceled);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> build_time = t1 - t0;
            SI_LOG_INFO(
                "Index built with max_data={}, num_data={}, build_time={} sec",
                index->maxDataPoints(),
                index->numData(),
                build_time.count() / 1000);
        }

        if (flags & VectorTestActions::SAVE)
        {
            SI_LOG_INFO("Save index with prefix {}", index_save_prefix);
            auto file_writer = IndexDataWriter(index_save_prefix);
            index->serialize(&file_writer);
            index->saveDataID(&file_writer);
        }

        if (flags & VectorTestActions::LOAD)
        {
            auto file_reader = IndexDataReader(index_save_prefix);
            SI_LOG_INFO("Load index with prefix {}", index_save_prefix);
            index->load(&file_reader);
            index->loadDataID(&file_reader);
            auto usage = index->getResourceUsage();
            SI_LOG_INFO(
                "Index {} memory_usage_bytes={}, disk_usage_bytes={}",
                enumToString(index_type),
                usage.memory_usage_bytes,
                usage.disk_usage_bytes);
        }

        std::shared_ptr<DenseBitmap> valid_ids{nullptr};
        if (filter_out_mod > 0 || filter_keep_min >= 0 || filter_keep_max >= 0)
        {
            SI_LOG_INFO(
                "Filtering out ids which mod {}=0, keep id between {} and {}",
                filter_out_mod,
                filter_keep_min,
                filter_keep_max);
            valid_ids
                = std::make_shared<DenseBitmap>(max_points + id_offset, true);
            for (size_t i = id_offset; i < max_points + id_offset; ++i)
            {
                bool keep = !((filter_out_mod > 0 && i % filter_out_mod == 0))
                    && !(filter_keep_min >= 0 and i < filter_keep_min)
                    && !(filter_keep_max >= 0 and i > filter_keep_max);
                if (!keep)
                    valid_ids->unset(i);
            }
        }

        if (flags & VectorTestActions::SEARCH)
        {
            SI_LOG_INFO("Constructing queries");
            int i0 = 9;
            int i1 = 18;
            SI_THROW_IF_NOT(i1 < max_points, Search::ErrorCode::LOGICAL_ERROR);

            int nq = i1 - i0;
            auto query_dataset = data_reader.getData(i0, nq);

            SI_LOG_INFO("Performing ANN SEARCH");
            //query_dataset->Dump("query_dataset");

            QueryStats stats;
            std::shared_ptr<SearchResult> res;
            res = index->search(
                query_dataset,
                top_k,
                search_params,
                /* first_stage_only */ false,
                valid_ids.get(),
                &stats);
            size_t num_incorrect_dis = 0;
            size_t num_incorrect_top1 = 0;
            float exact_dist = MetricToExactDist[metric];
            float dist_threshold = index_type_str.ends_with("PQ") ? 0.07 : 0.01;
            // TODO PQ with IP distance is quite troublesome: better not use it
            int max_errors = 1;
            bool metric_ip_cosine
                = metric == Metric::IP || metric == Metric::Cosine;
            if (index_type_str.ends_with("PQ") && metric_ip_cosine)
                max_errors = 4;
            auto nns = res->getResultIndices();
            auto dis = res->getResultDistances();
            std::string dis_str("\n");
            for (int j = 0; j < nq; ++j)
            {
                for (int k = 0; k < top_k; ++k)
                {
                    dis_str += std::to_string(dis[j * top_k + k]) + ", ";
                }
                dis_str += "\n";
            }
            SI_LOG_INFO("dis: {}", dis_str);
            for (int j = 0; j < nq; ++j)
            {
                bool valid = valid_ids == nullptr
                    || valid_ids->is_member(id_offset + i0 + j) == true;
                SI_LOG_INFO(
                    "Query={} FilterValid={} Top1 id={} dist={:.3f}",
                    i0 + j,
                    valid,
                    nns[j * top_k],
                    dis[j * top_k]);
                // skip invalid ids
                if (!valid)
                {
                    expect(nns[j * top_k] != i0 + j + id_offset);
                    SI_LOG_INFO(
                        "{} not in filter list, skip", i0 + j + id_offset);
                    continue;
                }
                if (std::abs(dis[j * top_k] - exact_dist) > dist_threshold)
                {
                    SI_LOG_INFO(
                        "dist: {}, exact_dist: {}, dist_threshold: {}",
                        dis[j * top_k],
                        exact_dist,
                        dist_threshold);
                }
                num_incorrect_dis
                    += (std::abs(dis[j * top_k] - exact_dist) > dist_threshold);
                num_incorrect_top1 += (nns[j * top_k] != i0 + j + id_offset);
                if (nns[j * top_k] != i0 + j + id_offset)
                {
                    SI_LOG_INFO(
                        "top1 not match : nns[j * top_k]: {} | (i0 + j + "
                        "id_offset): {}",
                        nns[j * top_k],
                        i0 + j + id_offset);
                }
            }
            SI_LOG_INFO(
                "num_incorrect_dis {}, num_incorrect_top1 {}",
                num_incorrect_dis,
                num_incorrect_top1);
            SI_LOG_INFO(
                "QueryStats: {}, num_incorrect_dis: {}, max_errors: {}",
                stats.toString(),
                num_incorrect_dis,
                max_errors);
            // allow for a few errors
            expect(num_incorrect_dis <= max_errors);
            if (data_dim >= 3)
            {
                // when dimension <= 3, ScaNN/IVFSQ gives correct distances but incorrect ids
                expect(num_incorrect_top1 <= max_errors);
                SI_LOG_INFO(
                    "num_incorrect_top1 {} max_errors {}",
                    num_incorrect_top1,
                    max_errors);
            }
        }
        printMemoryUsage();

        // run stress test and the end
        if ((flags & VectorTestActions::SEARCH)
            && !(flags & VectorTestActions::BUILD) && this->stress_threads > 0)
        {
            std::shared_mutex valid_ids_mutex;
            // reset valid_ids bitmap
            valid_ids
                = std::make_shared<DenseBitmap>(max_points + id_offset, true);

            // sample 100 random data for queries
            auto rand_queries = data_reader.sampleData(100);
            auto stress_run = [&](int worker_ind)
            {
                SI_LOG_INFO("Launching stress_test worker {}", worker_ind);
                auto t_start = std::chrono::high_resolution_clock::now();
                float last_log_sec = 0.0f;
                std::mt19937 gen(std::random_device{}());
                std::uniform_int_distribution<> dist;

                for (int i = 0;; ++i)
                {
                    if (worker_ind % 2 == 0)
                    {
                        std::shared_ptr<DenseBitmap> search_filter;
                        {
                            std::shared_lock<std::shared_mutex> lock(
                                valid_ids_mutex);
                            search_filter
                                = std::make_shared<DenseBitmap>(*valid_ids);
                        }
                        // search with random queries
                        auto rand_q = dist(gen) % rand_queries->numData();
                        auto query_dataset = std::make_shared<DataSet<T>>(
                            (*rand_queries)[rand_q], 1, this->data_dim);
                        auto res = index->search(
                            query_dataset,
                            top_k,
                            search_params,
                            /* first_stage_only */ false,
                            search_filter.get());
                    }
                    else if (worker_ind % 2 == 1)
                    {
                        // randomly flip 100 bits
                        {
                            std::unique_lock<std::shared_mutex> lock(
                                valid_ids_mutex);
                            for (int j = 0; j < 100; ++j)
                            {
                                auto r = dist(gen) % max_points + id_offset;
                                // flip the bit
                                if (valid_ids->is_member(r))
                                    valid_ids->unset(r);
                                else
                                    valid_ids->set(r);
                            }
                        }
                        // sleep for 20ms
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(10));
                    }

                    if (i % 50 == 0)
                    {
                        auto t_now = std::chrono::high_resolution_clock::now();
                        auto time_sec
                            = std::chrono::duration_cast<std::chrono::seconds>(
                                  t_now - t_start)
                                  .count();
                        // print log every 5 seconds
                        if (time_sec - last_log_sec >= 5)
                        {
                            SI_LOG_INFO(
                                "Stress test worker {} running for {} seconds, "
                                "loop_i={}",
                                worker_ind,
                                time_sec,
                                i);
                            last_log_sec = time_sec;
                        }
                        // exit from the loop if time is up
                        if (time_sec > this->stress_sec)
                            break;
                    }
                }
            };
            // launch threads to run stress_run, and wait for them to join
            std::vector<std::thread> work_threads;
            for (int t_ind = 0; t_ind < stress_threads; ++t_ind)
                work_threads.emplace_back(stress_run, t_ind);
            for (auto & t : work_threads)
                t.join();
        }
    }
};


class MemoryMonitor
{
public:
    MemoryMonitor(int sec = -1) :
        monitoring_thread_running(false), interval_sec(sec)
    {
    }

    void start()
    {
        if (interval_sec < 0)
            return;
        monitoring_thread_running.store(true);
        monitoring_thread = std::thread(&MemoryMonitor::monitor, this);
    }

    void abort()
    {
        if (interval_sec < 0)
            return;
        monitoring_thread_running.store(false);
        monitoring_thread.join();
    }

private:
    void monitor()
    {
        while (monitoring_thread_running.load())
        {
            print_memory_usage();
            std::this_thread::sleep_for(std::chrono::seconds(interval_sec));
        }
    }

    void print_memory_usage()
    {
        std::ifstream statm_file("/proc/self/statm");
        std::string line;
        std::getline(statm_file, line);
        std::istringstream iss(line);
        long long pages;
        iss >> pages; // Read the first value, which is the total program size in pages

        // Calculate memory usage in bytes
        long long memory_usage_mb = (pages * sysconf(_SC_PAGE_SIZE)) >> 20;
        SI_LOG_INFO(
            "Current statm memory usage: {} MB, RSS usage: {}",
            memory_usage_mb,
            getRSSUsage());
    }

    int interval_sec;
    std::atomic<bool> monitoring_thread_running;
    std::thread monitoring_thread;
};

int main(int argc, char * argv[])
{
#ifndef MYSCALE_MODE
    // initialize logging levels from environment variables
    spdlog::cfg::load_env_levels();
#endif

    using namespace boost::ut;

    argparse::ArgumentParser program("vector_index_test");
    program.add_argument("--metric")
        .default_value(std::string("L2"))
        .help("vector search metric: L2|IP|Cosine");
    program.add_argument("--index_types")
        .nargs(argparse::nargs_pattern::any)
        .help("Index types to run tests. Run all tests if not specified");
    program.add_argument("--num_build_threads")
        .default_value(0)
        .help("number of threads for building the index")
        .scan<'i', int>();
    program.add_argument("--filter_out_mod")
        .default_value(0)
        .help("filter out every Nth data point")
        .scan<'i', int>();
    program.add_argument("--filter_keep_min")
        .default_value(-1)
        .help("if >=0, only keep data points with id >= filter_keep_min")
        .scan<'i', int>();
    program.add_argument("--filter_keep_max")
        .default_value(-1)
        .help("if >=0, only keep data points with id <= filter_keep_max")
        .scan<'i', int>();
    program.add_argument("--data_dim")
        .default_value(128)
        .help("data dimension")
        .scan<'i', int>();
    program.add_argument("--data_dim_num_copy")
        .default_value(1)
        .help("data dimension")
        .scan<'i', int>();
    program.add_argument("--num_data")
        .default_value(2000)
        .help("data dimension")
        .scan<'i', int>();
    program.add_argument("--top_k")
        .default_value(5)
        .help("topK")
        .scan<'i', int>();
    program.add_argument("--use_default_params")
        .default_value(1)
        .help("use default params")
        .scan<'i', int>();
    program.add_argument("--scann_children_per_level")
        .default_value(std::string(""))
        .help("ints separate by _, say 10_10, empty for default value");
    program.add_argument("--scann_query_threads")
        .default_value(-1)
        .help("query threads for ScaNN index")
        .scan<'i', int>();
    program.add_argument("--scann_l_search")
        .default_value(-1)
        .help("l_search for ScaNN index (use -1 for default value)")
        .scan<'i', int>();
    program.add_argument("--scann_l_search_ratio")
        .default_value(-1.0f)
        .help("l_search for ScaNN index (use -1 for default value)")
        .scan<'g', float>();
    program.add_argument("--scann_num_reorder")
        .default_value(-1)
        .help("num_reorder for ScaNN index (use -1 for default value)")
        .scan<'i', int>();
    program.add_argument("--scann_dims_per_block")
        .default_value(-1)
        .help("dims_per_block for ScaNN index (use -1 for default value)")
        .scan<'i', int>();
    program.add_argument("--scann_aq_threshold")
        .default_value(-1.0f)
        .help("aq_threshold for ScaNN index (use -1 for default value)")
        .scan<'g', float>();
    program.add_argument("--scann_build_hashed_dataset_by_token")
        .default_value(0)
        .help("use hashed_dataset_by_token for ScaNN index (0 by default)")
        .scan<'i', int>();
    program.add_argument("--memory_monitor_sec")
        .default_value(-1)
        .help("memory monitoring for every k second")
        .scan<'i', int>();
    program.add_argument("--stress_threads")
        .default_value(-1)
        .help("parallel threads for stress test")
        .scan<'i', int>();
    program.add_argument("--rand_seed")
        .default_value(1248)
        .help("random seed for data generation")
        .scan<'i', int>();
    program.add_argument("--stress_sec")
        .default_value(60)
        .help("stress test seconds, default 60sec")
        .scan<'i', int>();
    program.add_argument("--abort_sec")
        .default_value(-1)
        .help("index->abort seconds, default -1 (inactive), 0: Ctrl-C")
        .scan<'i', int>();
    program.add_argument("--check_build_canceled_sec")
        .default_value(-1)
        .help("check_build_canceled wait seconds, default -1 (inactive)")
        .scan<'i', int>();
    program.add_argument("--abort_sec_stage")
        .default_value(0)
        .help("stage for index->abort, default 0: first stage")
        .scan<'i', int>();

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error & err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto mem_monitor_sec = program.get<int>("--memory_monitor_sec");
    MemoryMonitor memory_monitor(mem_monitor_sec);
    memory_monitor.start();
    OnExit abort_monitor([&memory_monitor]() { memory_monitor.abort(); });

    VectorIndexTester tester;

    std::string metric_str = program.get<std::string>("--metric");
    SI_THROW_IF_NOT_FMT(
        findEnumByName(metric_str, tester.metric),
        ErrorCode::UNSUPPORTED_PARAMETER,
        "Unsupported metric %s",
        metric_str.c_str());


    tester.max_points = program.get<int>("--num_data");
    tester.data_dim = program.get<int>("--data_dim");
    tester.data_dim_num_copy = program.get<int>("--data_dim_num_copy");
    tester.top_k = program.get<int>("--top_k");
    tester.scann_l_search = program.get<int>("--scann_l_search");
    tester.scann_l_search_ratio = program.get<float>("--scann_l_search_ratio");
    tester.scann_num_reorder = program.get<int>("--scann_num_reorder");
    tester.scann_dims_per_block = program.get<int>("--scann_dims_per_block");
    tester.scann_aq_threshold = program.get<float>("--scann_aq_threshold");
    tester.scann_build_hashed_dataset_by_token
        = program.get<int>("--scann_build_hashed_dataset_by_token");
    tester.scann_children_per_level
        = program.get<std::string>("--scann_children_per_level");
    tester.num_build_threads = program.get<int>("--num_build_threads");
    tester.filter_out_mod = program.get<int>("--filter_out_mod");
    tester.filter_keep_min = program.get<int>("--filter_keep_min");
    tester.filter_keep_max = program.get<int>("--filter_keep_max");
    tester.use_default_params = program.get<int>("--use_default_params");
    tester.stress_threads = program.get<int>("--stress_threads");
    tester.rand_seed = program.get<int>("--rand_seed");
    tester.stress_sec = program.get<int>("--stress_sec");
    tester.abort_sec = program.get<int>("--abort_sec");
    tester.check_build_canceled_sec
        = program.get<int>("--check_build_canceled_sec");
    auto abort_sec_stage = program.get<int>("--abort_sec_stage");
    auto index_str_vec = program.get<std::vector<std::string>>("--index_types");

    if (tester.abort_sec == 0)
    {
        // Block SIGINT/SIGUSR1 in the main thread and all future threads
        sigset_t signalSet;
        sigemptyset(&signalSet);
        sigaddset(&signalSet, SIGINT);
        sigaddset(&signalSet, SIGUSR1);
        pthread_sigmask(SIG_BLOCK, &signalSet, nullptr);
    }

    if (metric_str == "Hamming" || metric_str == "Jaccard")
    {
        for (auto index_type : BINARY_VECTOR_INDEX_TEST_TYPES)
        {
            auto index_str = enumToString(index_type);
            // skip if the index_types argument is not empty and doesn't contain the index
            if (!index_str_vec.empty()
                && (std::find(
                        index_str_vec.begin(), index_str_vec.end(), index_str)
                    == index_str_vec.end()))
                continue;
            // build, save and then search
            int flags = VectorTestActions::CREATE | VectorTestActions::BUILD
                | VectorTestActions::SAVE | VectorTestActions::SEARCH;
            tester.abort_sec_set = abort_sec_stage == 0;
            tester.testBinaryVectorIndex(index_type, flags);
        }
    }
    else
    {
        for (auto index_type : FLOAT_VECTOR_INDEX_TEST_TYPES)
        {
            auto index_str = enumToString(index_type);
            // skip if the index_types argument is not empty and doesn't contain the index
            if (!index_str_vec.empty()
                && (std::find(
                        index_str_vec.begin(), index_str_vec.end(), index_str)
                    == index_str_vec.end()))
                continue;
            // skip binary index for now
            if (index_str.starts_with("Binary"))
                continue;

            SI_LOG_INFO(
                "## Test creating & building & searching & saving index {}",
                enumToString(index_type));
            test("create_build_search_save_index_" + index_str) = [&]
            {
                // build, save and then search
                int flags = VectorTestActions::CREATE | VectorTestActions::BUILD
                    | VectorTestActions::SAVE | VectorTestActions::SEARCH;
                tester.abort_sec_set = abort_sec_stage == 0;
                tester.testFloatVectorIndex(index_type, flags);
            };

            SI_LOG_INFO(
                "## Test creating & loading & searching index {}",
                enumToString(index_type));
            test("create_load_search_index_" + index_str) = [&]
            {
                // load and then search
                int flags = VectorTestActions::CREATE | VectorTestActions::LOAD
                    | VectorTestActions::SEARCH;
                tester.abort_sec_set = abort_sec_stage == 1;
                tester.testFloatVectorIndex(index_type, flags);
            };
        }
    }
}
