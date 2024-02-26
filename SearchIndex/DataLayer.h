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
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <span>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <SearchIndex/Common/DenseDataLayer.h>
#include <SearchIndex/Common/IndexSourceData.h>
#include <SearchIndex/Common/Utils.h>
#include <faiss/utils/distances.h>

namespace Search
{

/// @brief Dense data layer backed by memory.
template <typename T>
class DenseMemoryDataLayer : public DenseDataLayer<T>
{
public:
    using DataChunk = DataSet<T>;
    using IStreamPtr = typename DenseDataLayer<T>::IStreamPtr;
    using OStreamPtr = typename DenseDataLayer<T>::OStreamPtr;
    bool use_lazy_decode = false;
    MemoryManager * mem_manager;

    DenseMemoryDataLayer(
        size_t max_data_,
        size_t data_dim_,
        bool init_data = false) :
        DenseDataLayer<T>(max_data_, data_dim_)
    {
        SI_LOG_INFO(
            "Creating DenseMemoryDataLayer data_dim={} "
            "use_lazy_decode={}",
            data_dim_,
            use_lazy_decode);

        mem_manager = nullptr;
        checkAvailableMemory(this->max_data * this->data_dim * sizeof(T));
        if (init_data)
        {
            data.resize(this->max_data * this->data_dim);
            // the data has already been initialized
            this->data_num = max_data_;
        }
    }

    virtual void remove() override
    {
        data.clear();
        this->max_data = 0;
        this->data_num = 0;
    }

    const T * getDataPtrImpl(idx_t idx) const
    {
        return &data[idx * this->data_dim];
    }

    virtual const T * getDataPtr(idx_t idx) const override
    {
        return getDataPtrImpl(idx);
    }

    virtual T * getDataPtr(idx_t idx) override
    {
        return const_cast<T *>(getDataPtrImpl(idx));
    }

    virtual void load(IStreamPtr reader, size_t /* num_data */ = -1) override
    {
        size_t npts, dim;
        std::visit(
            [&](auto && r) { load_bin_from_reader(*r, data, npts, dim); },
            reader);
        this->data_num = npts;
    }

    virtual size_t serialize(OStreamPtr writer) override
    {
        size_t checksum;
        std::visit(
            [&](auto && w)
            {
                save_bin_with_writer(
                    *w,
                    data.data(),
                    this->dataNum(),
                    this->dataDimension(),
                    &checksum);
            },
            writer);
        SI_LOG_INFO(
            "DenseMemoryDataLayer::serialize checksum={} dataDimension={}",
            checksum,
            this->dataDimension());
        return checksum;
    }

protected:
    virtual void addDataImpl(DataChunk & chunk) override
    {
        checkAvailableMemory(
            chunk.numData() * chunk.dimension() * sizeof(T));
        data.insert(data.end(), chunk.getData(), chunk.dataEnd());
        this->data_num += chunk.numData();

        SI_THROW_IF_NOT(
            this->data_num * this->data_dim * sizeof(float)
                == data.size() * sizeof(float),
            ErrorCode::LOGICAL_ERROR);
    }

    std::vector<T> data;
};


}
