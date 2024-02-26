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

#pragma once
#include <shared_mutex>
#include <SearchIndex/Common/IDSelector.h>
#include <SearchIndex/Common/IndexDataIO.h>
#include <SearchIndex/Common/IndexSourceData.h>
#include <SearchIndex/SearchIndexCommon.h>
#include <SearchIndex/SearchResult.h>

namespace Search
{

/** Status of Search Index
 *
 * Broadly, the statuses include INIT/BUILDING/SEALED. Building can be further
 * divided into TRAINING, TRAINED, ADDING, if supported by the index.
 *
 * If the index is loaded from disk, then status goes from INIT -> LOADING -> LOADED.
 */
enum class IndexStatus
{
    INIT,
    BUILDING,
    BUILD_TRAINING,
    BUILD_TRAINED,
    BUILD_ADDING,
    SEALED,
    LOADING,
    LOADED
};

/// When index is in ABORTING/ABORTED state, it will reject all build/load/search requests
enum class IndexAbortStatus
{
    NONE = 0,
    ABORTING = 1,
    ABORTED = 2
};

/**
 * @brief resource usage of index, such as memory and disk usage
*/
struct IndexResourceUsage
{
    // search resource usage
    size_t memory_usage_bytes{0};
    size_t disk_usage_bytes{0};
    // build resource usage
    size_t build_memory_usage_bytes{0};
    // build_disk_usage_bytes is zero for memory-only indexes (eg. FLAT, IVF, HNSW)
    size_t build_disk_usage_bytes{0};

    IndexResourceUsage & operator+=(const IndexResourceUsage & other)
    {
        memory_usage_bytes += other.memory_usage_bytes;
        disk_usage_bytes += other.disk_usage_bytes;
        build_memory_usage_bytes += other.build_memory_usage_bytes;
        build_disk_usage_bytes += other.build_disk_usage_bytes;
        return *this;
    }
};

/**
 * @brief version of index, such as 1.0.0, persisted by MyScale database.
 *
 * When index version doesn't match, index loading might fail.
*/
struct IndexVersion
{
    size_t major;
    size_t minor;
    size_t patch;

    bool operator==(const IndexVersion & other) const
    {
        return (major == other.major) && (minor == other.minor)
            && (patch == other.patch);
    }

    bool operator<=(const IndexVersion & other) const
    {
        if (major > other.major)
            return false;
        if (minor > other.minor)
            return false;
        if (patch > other.patch)
            return false;
        return true;
    }

    std::string toString()
    {
        return std::to_string(major) + "." + std::to_string(minor) + "."
            + std::to_string(patch);
    }

    static IndexVersion fromString(const std::string & str)
    {
        size_t pos = 0;
        std::vector<std::string> subs;
        while (pos < str.size())
        {
            size_t q = str.find(".", pos);
            if (q == std::string::npos)
            {
                q = str.size();
            }
            subs.emplace_back(str, pos, q - pos);
            pos = q + 1;
        }

        SI_THROW_IF_NOT_MSG(
            subs.size() == 3,
            ErrorCode::LOGICAL_ERROR,
            "Version string must have three numbers");
        return IndexVersion{
            std::stoul(subs[0]), std::stoul(subs[1]), std::stoul(subs[2])};
    }
};

using IDListPtr = typename std::shared_ptr<std::vector<idx_t>>;

/**
 * @brief purpose IDListSelector for SearchIndex
 *
 * @tparam IDS Underlying IDSelector, usually (compressed) bitmap
 * IDListSelector translates position => id with id_list, then check IDS::is_member.
 * It's used as a neutral choice when the index library doesn't have a proper
 * IDSelector abstraction (such as hnswlib/ScaNN).
 */
template <IDSelector IDS>
struct IDListSelector final : public AbstractIDSelector
{
    explicit IDListSelector(std::vector<idx_t> * id_list_, IDS * selector_) :
        id_list(id_list_), selector(selector_)
    {
    }

    /// Get the data_id, and check whether its in the selector
    bool is_member(idx_t pos) const override
    {
        return selector->is_member(id_list->at(pos));
    }

private:
    std::vector<idx_t> * id_list;
    IDS * selector;
};

/**
 * @brief SearchIndex is the base class for all search indexes.
*/
template <typename IS, typename OS, IDSelector IDS, DataType dataType>
class SearchIndex
{
public:
    using IndexDataType = typename DataTypeMap<dataType>::type;
    using UniqueLock = std::unique_lock<std::shared_mutex>;
    typedef void SearchParamsAdapter(Parameters & params);

    /// @brief numData() return value when id_list is not set.
    static const int64_t UNKNOWN_NUM_DATA = -1;

    /// @brief Name of the field for id_list.
    static inline const std::string DATA_ID_FIELD_NAME = "id_list";

    SearchIndex(const std::string & name_, size_t max_points_) :
        name(name_), max_points(max_points_), status(IndexStatus::INIT)
    {
    }

    /// @brief Name of the search index, useful for debugging purposes.
    virtual std::string getName() const { return name; }

    /// @brief Get the version of the index.
    virtual IndexVersion getVersion() const = 0;

    /// @brief Get current status of the index (e.g. building, loading, loaded, etc.)
    IndexStatus getStatus() const { return status; }

    /// @brief Return whether the index is ready for search
    virtual bool ready() const
    {
        IndexStatus s = getStatus();
        return s == IndexStatus::SEALED || s == IndexStatus::LOADED;
    }

    /// @brief Set new status and return current one
    IndexStatus setStatus(IndexStatus s) { return status.exchange(s); }

    // explicit SearchIndex(const Parameters& params, const char* disk_cache_path = nullptr);

    /**
     * @brief Build search index from data (e.g. from DB for File)
     *
     * Internally, build() calls train(), add() & seal().
     * @param num_threads number of threads used in building index (must be positive to take effect)
     */
    void build(
        IndexSourceDataReader<IndexDataType> * reader,
        int num_threads = 0,
        std::function<bool()> check_build_canceled = nullptr)
    {
        using DataChunk =
            typename IndexSourceDataReader<IndexDataType>::DataChunk;
        {
            /// set the building status
            UniqueLock mutation_lock(this->mutation_mutex);
            SIConfiguration::setCurrentThreadCheckAbortHandler(
                [this, &check_build_canceled]()
                { return this->checkAndAbort(check_build_canceled); });
            OnExit clear_check_abort(
                []()
                { SIConfiguration::clearCurrentThreadCheckAbortHandler(); });

            checkAndAbort();
            setStatus(IndexStatus::BUILDING);

            /// Extend DataID after every chunk read
            this->initDataID();
            reader->setAfterReadDataCallBack(
                [this](DataChunk * chunk)
                { this->extendDataID(chunk->numData(), chunk->getDataID()); });

            auto s = std::chrono::high_resolution_clock::now();
            SI_LOG_INFO("{} start building", this->getName());
            RECORD_MEMORY_USAGE("begin");
            buildImpl(reader, num_threads);
            RECORD_MEMORY_USAGE("buildImpl");
            auto diff = (std::chrono::high_resolution_clock::now() - s);
            [[maybe_unused]] double sec
                = static_cast<double>(diff.count() / 10000000) / 1e2;
            SI_LOG_INFO(
                "{} building finished, indexing_time={}s",
                this->getName(),
                sec);

            SI_THROW_IF_NOT_FMT(
                getStatus() == IndexStatus::SEALED,
                ErrorCode::LOGICAL_ERROR,
                "Index building status not successful: %s",
                enumToString(getStatus()).c_str());
        }

        if (this->load_after_build)
        {
            /// data already present, no need to provide the reader
            this->load(nullptr);
        }
    }

    /**
     * @brief train() collects stats from the data in preparation for add()
     *
     * For IVF, do clustering; for PQ/SQ, do quantization; for HNSW, no-op.
     */
    virtual void
    train(IndexSourceDataReader<IndexDataType> * reader, int num_threads = 0)
        = 0;

    /// @brief Add data to existing index
    virtual void
    add(IndexSourceDataReader<IndexDataType> * reader, int num_threads = 0)
        = 0;

    /** @brief Seal the current index.
     *
     * For many indexes this may be no-np, DiskANN may perform graph building & merging, etc.
     */
    virtual void seal(int num_threads) = 0;

    /** @brief Set search parameter adapter.
     *
     * @param adapter function object to change the search parameter
     * Adapter is useful for flat/normal search, or change search parameter for acceleration.
     */
    void setSearchParamsAdapter(SearchParamsAdapter * adapter)
    {
        search_params_adapter = adapter;
    }

    /// @brief Return index build progress, between 0 and 1.
    virtual float getBuildProgress()
    {
        switch (status)
        {
            case IndexStatus::INIT:
                return 0.0f;
            case IndexStatus::BUILDING:
            case IndexStatus::BUILD_TRAINING:
                return 0.1f;
            case IndexStatus::BUILD_TRAINED:
                return 0.2f;
            case IndexStatus::BUILD_ADDING:
                return 0.5f;
            case IndexStatus::SEALED:
            case IndexStatus::LOADED:
                return 1.0f;
            case IndexStatus::LOADING:
                return NAN;
        }
        /// All cases has been covered, so we remove the default case to avoid
        /// warning "default label in switch which covers all enumeration values"
    }

    virtual IndexResourceUsage getResourceUsage() const = 0;

    /**
     * @brief Abort the current building/loading process,
     *  and wait for current search() process to finish.
     *
     * In multi-threaded case, the caller should call this function
     * first before deleting the index files, to avoid issues of
     * a corrupted index state.
     */
    void abort()
    {
        /// set status to ABORTING for other threads to check and abort
        abort_status = IndexAbortStatus::ABORTING;
        /// obtain the lock and set ABORTED status
        UniqueLock lock(mutation_mutex);
        abort_status = IndexAbortStatus::ABORTED;
        SI_LOG_INFO(
            "{} status={} is aborted", getName(), enumToString(status.load()));
    }

    /// @brief Load search index from serialized data.
    void load(
        IndexDataReader<IS> * reader,
        std::function<bool()> check_build_canceled = nullptr)
    {
        UniqueLock mutation_lock(mutation_mutex);
        SIConfiguration::setCurrentThreadCheckAbortHandler(
            [this, &check_build_canceled]() { return this->checkAndAbort(check_build_canceled); });
        OnExit clear_check_abort(
            []() { SIConfiguration::clearCurrentThreadCheckAbortHandler(); });

        checkAndAbort();
        SI_LOG_INFO("{} starts loading", getName());
        RECORD_MEMORY_USAGE("before load");
        loadImpl(reader);
        RECORD_MEMORY_USAGE("after load");
    }

    /// @brief Serialize search index to storage system
    void serialize(IndexDataWriter<OS> * writer)
    {
        UniqueLock mutation_lock(mutation_mutex);
        checkAndAbort();
        serializeImpl(writer);
    }

    // @brief internal implmentation of serialize()
    virtual void serializeImpl(IndexDataWriter<OS> * writer) = 0;

    /// Return maximum number of data points in the index
    size_t maxDataPoints() const { return max_points; }

    /** Number of data points in the vector index
     *
     * @return size of id_list if it's valid, UNKNOWN otherwise
     */
    int64_t numData() const
    {
        return id_list ? id_list->size() : UNKNOWN_NUM_DATA;
    }

    /**
     * @brief Set the id_list manually
     *
     * throw an exception if id_list size doesn't match
     */
    void setDataID(IDListPtr new_id_list)
    {
        checkIDListLengthAndThrowError(new_id_list->size());
        id_list = new_id_list;
    }

    /// @brief Get current data id_list
    IDListPtr getDataID() { return id_list; }

    /// @brief Load data id_list from IndexDataReader
    virtual void loadDataID(IndexDataReader<IS> * reader)
    {
        auto istream = reader->getFieldDataInputStream(DATA_ID_FIELD_NAME);
        loadDataID(istream.get());
    }

    /// @brief Save data id_list to IndexDataWriter
    virtual void saveDataID(IndexDataWriter<OS> * writer)
    {
        auto ostream = writer->getFieldDataOutputStream(DATA_ID_FIELD_NAME);
        saveDataID(ostream.get());
    }

    /// @brief Load data id_list from input stream
    void loadDataID(IS * in_stream)
    {
        size_t len;
        in_stream->read(reinterpret_cast<char *>(&len), sizeof(size_t));
        /// length must match
        checkIDListLengthAndThrowError(len);
        if (!id_list)
        {
            id_list = std::make_shared<std::vector<idx_t>>(len);
        }
        in_stream->read(
            reinterpret_cast<char *>(&id_list->at(0)), len * sizeof(idx_t));
    }

    /// @brief Write data id_list to output stream
    void saveDataID(OS * out_stream) const
    {
        if (!id_list)
        {
            SI_THROW_MSG(ErrorCode::LOGICAL_ERROR, "id_list not initialized");
        }
        size_t len = id_list->size();
        out_stream->write(reinterpret_cast<char *>(&len), sizeof(size_t));
        out_stream->write(
            reinterpret_cast<char *>(&id_list->at(0)),
            id_list->size() * sizeof(idx_t));
    }

    virtual ~SearchIndex() = default;

protected:
    /// @brief Name of the index for introspective purposes
    std::string name;

    /// @brief Maximum number of data points in the index
    size_t max_points;

    /// @brief External ID of the data, length is number of data entries if initialized
    IDListPtr id_list;

    /// @brief Current status of the index
    std::atomic<IndexStatus> status;

    /// @brief abort_status of the index (i.e. whether it's aborting/aborted)
    std::atomic<IndexAbortStatus> abort_status{IndexAbortStatus::NONE};

    /**
     * @brief Mutex for protecting internal state of the search-index
     *
     * Read operations (e.g. search) can be performed concurrently by hold a shared lock
    */
    std::shared_mutex mutation_mutex;

    /**
     * @brief When adapter is not null, it changes the search params adaptively
     */
    SearchParamsAdapter * search_params_adapter{nullptr};

    /**
     * @brief Whether to perform loading after building
     *
     * Useful for disk-based indices, false by default
     */
    bool load_after_build{false};

    virtual void
    buildImpl(IndexSourceDataReader<IndexDataType> * reader, int num_threads)
        = 0;

    virtual void loadImpl(IndexDataReader<IS> * reader) = 0;

    /** If abort_status == ABORTING/ABORTED, set it to ABORTED and throw exception,
     *  causing the current action to abort and lock released.
     */
    void checkAndAbort(std::function<bool()> abort_checker = nullptr)
    {
        if (abort_status != IndexAbortStatus::NONE
            || (abort_checker && abort_checker()))
        {
            SI_LOG_ERROR("Aborting the SearchIndex now");
            /// if status is ABORTING, set it to ABORTED
            abort_status = IndexAbortStatus::ABORTED;

            /// throw an exception to abort current action
            SI_THROW_MSG(
                ErrorCode::ABORTED, "index is aborted, abort all actions");
        }
    }

    /// @brief Call this function first in search to adapt the parameters.
    void adaptSearchParams(Parameters & params)
    {
        if (search_params_adapter)
        {
            search_params_adapter(params);
        }
    }

    /// @brief Initialize id_list
    void initDataID() { id_list = std::make_shared<std::vector<idx_t>>(); }

    /// @brief Return current id_list size. If it's null, return -1
    int64_t getDataSize() const { return id_list ? id_list->size() : -1; }

    /**
     * @brief Add new data ids to existing id_list
     *
     * Increment the id_list in serial order if new_id_list is null
     */
    void extendDataID(size_t n, const idx_t * new_id_list = nullptr)
    {
        idx_t id_end = id_list->empty() ? -1 : id_list->back();
        id_list->reserve(n);
        for (size_t i = 0; i < n; ++i)
        {
            idx_t idx = new_id_list ? new_id_list[i] : id_end + i + 1;
            id_list->push_back(idx);
        }
    }

    /**
     * @brief Translate from position => data_id, update inplace if out_list is null
     *
     * If position < 0, directly write it to output.
     */
    void translateDataID(
        idx_t * pos_list, size_t n, idx_t * out_list = nullptr) const
    {
        idx_t * res = out_list ? out_list : pos_list;
        for (size_t i = 0; i < n; ++i)
        {
            auto pos = pos_list[i];
            if (pos >= 0)
                SI_THROW_IF_NOT_FMT(
                    pos < numData(),
                    ErrorCode::LOGICAL_ERROR,
                    "pos (%ld) should be less than numData() (%ld)",
                    pos,
                    numData());
            res[i] = pos >= 0 ? id_list->at(pos) : pos;
        }
    }

    /// @brief Check whether IDList size is equal to len, throw an exception if not.
    bool checkIDListLengthAndThrowError(idx_t len)
    {
        if (id_list && static_cast<idx_t>(id_list->size()) != len)
        {
            SI_LOG_FATAL(
                "id_list size doesn't match: {} != {}", id_list->size(), len);
            SI_THROW_MSG(
                ErrorCode::LOGICAL_ERROR, "id_list size doesn't match");
        }
        return true;
    }
};
}
