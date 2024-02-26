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

#include <any>
#include <concepts>
#include <iostream>
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <SearchIndex/Common/IndexDataIO.h>
#include <SearchIndex/Common/Utils.h>
#include <magic_enum/magic_enum.hpp>

namespace Search
{

enum class IndexType
{
    IVFFLAT,
    IVFPQ,
    IVFSQ,
    FLAT,
    HNSWfastFLAT,
    HNSWfastPQ,
    HNSWfastSQ,
    HNSWFLAT,
    HNSWPQ,
    HNSWSQ,
    SCANN,
    BinaryIVF,
    BinaryHNSW,
    BinaryFLAT,
};

enum class Metric
{
    L2,
    IP,
    Cosine,
    Hamming,
    Jaccard
};

/// Single/Batch Query Aggregate Stats
struct QueryStats
{
    unsigned num_queries = 0;
    float total_us = 0; // total time to process query in micros
    unsigned n_ios = 0; // total # of IOs issued
    unsigned n_cmps = 0; // # cmps
    unsigned n_cache_hits = 0; // # cache_hits
    unsigned n_hops = 0; // # search hops

    /// list all fields for python processing
    inline static const std::vector<std::string> fields = {
        "num_queries", "total_us", "n_ios", "n_cmps", "n_cache_hits", "n_hops"};

    std::string toString()
    {
        json j
            = {{"num_queries", num_queries},
               {"total_us", total_us},
               {"n_ios", n_ios},
               {"n_cmps", n_cmps},
               {"n_cache_hits", n_cache_hits},
               {"n_hops", n_hops}};
        return j.dump();
    }

    QueryStats & operator+=(const QueryStats & rhs)
    {
        this->num_queries += rhs.num_queries;
        this->total_us += rhs.total_us;
        this->n_ios += rhs.n_ios;
        this->n_cmps += rhs.n_cmps;
        this->n_cache_hits += rhs.n_cache_hits;
        this->n_hops += rhs.n_hops;
        return *this;
    }
};


/// Different data types supported by search index

enum class DataType
{
    FloatVector = 0,
    Int8Vector,
    UInt8Vector,
    BinaryVector, // BinaryVector is represented as String in Database
};

//  DataTypeMap maps DataType enum values actual types
template <DataType>
struct DataTypeMap;

template <>
struct DataTypeMap<DataType::FloatVector>
{
    using type = float;
    using distance_type = float;
    static const size_t bits = 32;
};

template <>
struct DataTypeMap<DataType::UInt8Vector>
{
    using type = uint8_t;
    using distance_type = int32_t;
    static const size_t bits = 8;
};

template <>
struct DataTypeMap<DataType::Int8Vector>
{
    using type = int8_t;
    using distance_type = int32_t;
    static const size_t bits = 8;
};

template <>
struct DataTypeMap<DataType::BinaryVector>
{
    using type = bool;
    using distance_type = int32_t;
    static const size_t bits = 1;
};

/// Field meta data parse & dump

struct IndexFieldMetaIO
{
    static inline IndexDataFieldMeta parse(const json & data)
    {
        return IndexDataFieldMeta{data["name"], data["size"], data["checksum"]};
    }

    static inline std::string dump(const IndexDataFieldMeta & m)
    {
        json data
            = {{"name", m.name}, {"size", m.size}, {"checksum", m.checksum}};
        // return a single json line
        return data.dump();
    }
};

/// Concept for ID filter (borrowed from faiss)

template <typename T>
concept IDSelector = requires(T s, idx_t id)
{
    {
        s.is_member(id)
        } -> std::same_as<bool>;
};


/// Index parameter types and parsing functions

using ParametersMap = std::unordered_map<std::string, std::string>;

struct Parameters : public ParametersMap
{
    Parameters() { }

    Parameters(std::initializer_list<ParametersMap::value_type> init) :
        ParametersMap(init)
    {
    }

    template <typename T>
    void setParam(const std::string & name, T value)
    {
        (*this)[name] = std::to_string(value);
    }

    template <>
    void setParam(const std::string & name, bool value)
    {
        // store bool value as binary integer
        (*this)[name] = value ? "1" : "0";
    }

    template <>
    void setParam(const std::string & name, std::string value)
    {
        (*this)[name] = std::move(value);
    }

    template <>
    void setParam(const std::string & name, std::vector<std::string> value)
    {
        // concatenate elements with delimiter _
        (*this)[name] = vectorToString(value, "_");
    }

    template <typename T>
    T getParam(const std::string & name, T default_value) const
    {
        auto it = this->find(name);
        if (it == this->end())
        {
            return default_value;
        }
        return parseParamValue<T>(it->second);
    }

    template <typename T>
    T extractParam(
        const std::string & name, T default_value, bool remove = true)
    {
        auto it = this->find(name);
        if (it == this->end())
        {
            return default_value;
        }
        std::string s = it->second;
        if (remove)
        {
            this->erase(it);
        }
        return parseParamValue<T>(s);
    }

    std::string toString() const
    {
        std::string s;
        for (auto & it : *this)
        {
            s += " " + it.first + "=" + it.second;
        }
        return s;
    }

private:
    // no default parsing function
    template <typename T>
    T parseParamValue(const std::string & s) const = delete;

    template <>
    std::string parseParamValue<std::string>(const std::string & s) const
    {
        return s;
    }

    template <>
    size_t parseParamValue<size_t>(const std::string & s) const
    {
        return std::stoul(s);
    }

    template <>
    int parseParamValue<int>(const std::string & s) const
    {
        return std::stoi(s);
    }

    template <>
    uint32_t parseParamValue<uint32_t>(const std::string & s) const
    {
        unsigned long value = std::stoul(s);

        if (value > std::numeric_limits<uint32_t>::max())
        {
            throw std::out_of_range("Value is out of range for uint32_t");
        }
        return static_cast<uint32_t>(value);
    }

    template <>
    float parseParamValue<float>(const std::string & s) const
    {
        return std::stof(s);
    }

    template <>
    bool parseParamValue<bool>(const std::string & s) const
    {
        int value = std::stoi(s);
        SI_THROW_IF_NOT_FMT(
            value == 0 || value == 1,
            ErrorCode::UNSUPPORTED_PARAMETER,
            "boolean value must be 0 or 1: %d",
            value);
        return value == 1;
    }

    template <>
    std::vector<int32_t>
    parseParamValue<std::vector<int32_t>>(const std::string & s) const
    {
        std::vector<int32_t> arr;
        if (s.empty())
            return arr;

        // split string with _ as delimiter and convert each element to uint32_t
        size_t pos = 0, st = 0;
        while (true)
        {
            pos = s.find("_", st);
            auto subs = pos == std::string::npos ? s.substr(st)
                                                 : s.substr(st, pos - st);
            arr.push_back(std::stoi(subs));
            if (pos == std::string::npos)
                break;
            st = pos + 1;
        }
        return arr;
    }
};


inline void raiseErrorOnUnknownParams(const Parameters & params)
{
    if (!params.empty())
    {
        SI_THROW_MSG(
            ErrorCode::UNSUPPORTED_PARAMETER,
            "Unknown parameter:" + params.toString());
    }
}

}
