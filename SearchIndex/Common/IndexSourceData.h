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

#include <limits>
#include <memory>

#include "Utils.h"

// This file is to be included by both MyScaleSearchIndex and the individual vector search libraries.
namespace Search
{

using idx_t = int64_t;

// Dataset of fixed arrays
template <typename T>
class DataSet
{
public:
    typedef std::function<void(void)> Deleter;

    DataSet(
        const T * data_,
        size_t data_num_,
        size_t data_dim_,
        Deleter deleter_ = nullptr) :
        data(data_), data_num(data_num_), data_dim(data_dim_), deleter(deleter_)
    {
        if (!data_)
        {
            data_container.resize(data_num_ * singleVectorSizeInByte());
        }
    }

    void setDataID(const idx_t * id_list_, Deleter id_deleter_ = nullptr)
    {
        id_list = id_list_;
        id_deleter = id_deleter_;
    }

    size_t singleVectorSizeInByte() const;

    std::shared_ptr<DataSet<T>> normalize(bool reuse = false, float norm = 1)
    {
        std::shared_ptr<DataSet<T>> ret;
        T * data_normalized;
        if (reuse)
        {
            data_normalized = const_cast<T *>(data);
            ret = std::make_shared<DataSet<T>>(
                data_normalized, data_num, data_dim, deleter);
            deleter = nullptr;
            if (id_list != nullptr)
            {
                ret->setDataID(id_list, id_deleter);
                id_deleter = nullptr;
            }
        }
        else
        {
            data_normalized = new T[data_num * data_dim];
            ret = std::make_shared<DataSet<T>>(
                data_normalized,
                data_num,
                data_dim,
                [=]() { delete[] data_normalized; });
            if (id_list != nullptr)
            {
                auto id_list_cpy = new idx_t[data_num];
                std::copy(id_list, id_list + data_num, id_list_cpy);
                ret->setDataID(id_list_cpy, [=]() { delete[] id_list_cpy; });
            }
        }
        for (size_t idx = 0; idx < data_num; idx++)
        {
            float sum = 0;
            auto origin = (*this)[idx];
            for (size_t d = 0; d < data_dim; d++)
            {
                sum += origin[d] * origin[d];
            }
            if (sum < std::numeric_limits<float>::epsilon())
                continue;
            sum = std::sqrt(sum);
            auto target = data_normalized + idx * data_dim;
            for (size_t d = 0; d < data_dim; d++)
            {
                target[d] = origin[d] / sum * norm;
            }
        }
        return ret;
    }

    const T * operator[](size_t idx) const { return data + idx * dimension(); }

    const T * getData() const
    {
        return reinterpret_cast<const T *>(
            data ? data
                 : reinterpret_cast<T *>(
                     const_cast<uint8_t *>(data_container.data())));
    }

    const T * dataEnd() const { return getData() + numData() * dimension(); }

    const idx_t * getDataID() const { return id_list; }

    size_t numData() const { return data_num; }

    size_t dimension() const { return data_dim; }

    std::shared_ptr<DataSet<T>> padDataDimension(size_t target_dim) const
    {
        // must pad to a larger dimension
        SI_THROW_IF_NOT_FMT(
            dimension() < target_dim,
            ErrorCode::LOGICAL_ERROR,
            "padDataDimension error: target_dim %lu <= dimension %lu",
            target_dim,
            dimension());
        T * data_padded = new T[numData() * target_dim];
        for (size_t i = 0; i < numData(); i++)
        {
            T * target_row = data_padded + i * target_dim;
            std::copy((*this)[i], (*this)[i + 1], target_row);
            std::fill(target_row + dimension(), target_row + target_dim, 0);
        }
        auto ret = std::make_shared<DataSet<T>>(
            data_padded,
            numData(),
            target_dim,
            [=]() { delete[] data_padded; });
        if (id_list != nullptr)
        {
            idx_t * id_list_cpy = new idx_t[numData()];
            std::copy(id_list, id_list + numData(), id_list_cpy);
            ret->setDataID(id_list_cpy, [=]() { delete[] id_list_cpy; });
        }
        return ret;
    }

    T * getRawData() const
    {
        if (!data)
        {
            return reinterpret_cast<T *>(
                const_cast<uint8_t *>(data_container.data()));
        }
        else
        {
            return reinterpret_cast<T *>(const_cast<T *>(data));
        }
    }

    ~DataSet()
    {
        if (deleter)
            deleter();
        if (id_deleter)
            id_deleter();
    }

public:
    std::vector<uint8_t> data_container;

private:
    const T * data{nullptr};
    const idx_t * id_list{nullptr};
    size_t data_num;
    size_t data_dim;
    Deleter deleter{nullptr};
    Deleter id_deleter{nullptr};
};

/// DataReader reads data from Database columns or files to build index
template <typename T>
class IndexSourceDataReader
{
public:
    using DataChunk = DataSet<T>;
    typedef std::function<void(DataChunk *)> ChunkCallback;

    /** Sample some data for index training (e.g. clustering).
     *
     * SampleData doesn't maintain any internal buffers
     */
    virtual std::shared_ptr<DataChunk> sampleData(size_t n) = 0;

    /** Read data of size n.
     *
     * @return chunk of data read. If no more data can be read, return a null pointer.
     * readData() maintains an internal buffer.
     */
    std::shared_ptr<DataChunk> readData(size_t n)
    {
        auto chunk = this->readDataImpl(n);
        if (after_read_data_cb)
        {
            after_read_data_cb(chunk.get());
        }
        return chunk;
    }

    /// Number of data points already read.
    virtual size_t numDataRead() const = 0;

    // Dimension of the data vectors
    virtual size_t dataDimension() const = 0;

    size_t dataSize() const { return sizeof(T) * dataDimension(); }

    /// This callback is executed when data chunk is read out
    void setAfterReadDataCallBack(const ChunkCallback & cb)
    {
        after_read_data_cb = cb;
    }

    /// File-like read() interface
    void read(char * s, std::streamsize count)
    {
        SI_THROW_IF_NOT_MSG(
            count % dataSize() == 0,
            ErrorCode::LOGICAL_ERROR,
            "Only support reading whole rows of data");

        size_t num_rows = count / dataSize();
        auto chunk = readData(num_rows);
        last_read_count = chunk->numData() * dataSize();
        memcpy(s, chunk->getData(), last_read_count);
    }

    /// Return number of bytes in last read
    size_t gcount() { return last_read_count; }

    /// Provides file-like seek interface (to read data again)
    virtual void seekg(std::streamsize offset, std::ios::seekdir dir) = 0;

    /// Returns whether the reader has exhausted all data
    virtual bool eof() = 0;

    virtual ~IndexSourceDataReader() = default;

protected:
    /// Actual readData procedure, to be implemented by subclasses
    virtual std::shared_ptr<DataChunk> readDataImpl(size_t n) = 0;

    size_t last_read_count{0};

    ChunkCallback after_read_data_cb{nullptr};
};


template <typename T>
class IndexSourceDataPartReader : public IndexSourceDataReader<T>
{
public:
    using DataChunk = DataSet<T>;
    typedef std::function<void(DataChunk *)> ChunkCallback;

    IndexSourceDataPartReader(
        IndexSourceDataReader<T> * data_source_,
        size_t offset_,
        size_t count_) :
        data_source(data_source_), offset(offset_), count(count_)
    {
        SI_THROW_IF_NOT_MSG(
            offset == data_source->numDataRead(),
            ErrorCode::LOGICAL_ERROR,
            "offset != data_source->numDataRead()");
    }

    // Sample data starting from current offset
    virtual std::shared_ptr<DataChunk> sampleData(size_t n) override
    {
        return data_source->sampleData(n);
    }

    /// Number of data points already read.
    virtual size_t numDataRead() const override
    {
        return data_source->numDataRead() - offset;
    }

    // Dimension of the data vectors
    virtual size_t dataDimension() const override
    {
        return data_source->dataDimension();
    }

    /// Provides file-like seek interface (to read data again)
    virtual void
    seekg(std::streamsize /*offset*/, std::ios::seekdir /*dir*/) override
    {
        SI_THROW_MSG(
            ErrorCode::NOT_IMPLEMENTED, "seekg not supported for part reader");
    }

    /// Returns whether the part reader has exhausted all data
    virtual bool eof() override
    {
        return data_source->eof() || numDataRead() >= count;
    }

    virtual ~IndexSourceDataPartReader() override = default;

protected:
    /// Actual readData procedure, to be implemented by subclasses
    virtual std::shared_ptr<DataChunk> readDataImpl(size_t n) override
    {
        n = std::min(n, count - numDataRead());
        return data_source->readData(n);
    }

private:
    IndexSourceDataReader<T> * data_source;
    size_t offset;
    size_t count;
};

}
