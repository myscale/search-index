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

#include <SearchIndex/VectorSearch.h>
#include "SearchIndexCommon.h"

#define FLOAT_VECTOR_METRIC_TYPE_PARAMETER  R"("metric_type": {"type": "string", "case_sensitive": "false", "range":[], "candidates":["L2", "Cosine", "IP"]})"
#define BINARY_VECTOR_METRIC_TYPE_PARAMETER R"("metric_type": {"type": "string", "case_sensitive": "false", "range":[], "candidates":["Hamming", "Jaccard"]})"

#define VECTOR_INDEX_IVF_BASE_PARAMETER     R"("ncentroids": {"type": "int", "case_sensitive": "false", "range":[1, 1048576], "candidates":[]}, "nprobe": {"type": "int", "case_sensitive": "false", "range":[1,1048576], "candidates":[]})"
#define VECTOR_INDEX_HNSW_BASE_PARAMETER    R"("m": {"type": "int", "case_sensitive": "false", "range":[8, 128], "candidates":[]}, "ef_c": {"type": "int", "case_sensitive": "false", "range":[16, 1024], "candidates":[]}, "ef_s": {"type": "int", "case_sensitive": "false", "range":[16,1024], "candidates":[]})"

#define VECTOR_SEARCH_ALPHA_PARAMETER   R"("alpha": {"type": "float", "case_sensitive": "false", "range":[1,4], "candidates":[]})"
#define VECTOR_INDEX_IVF_M_PARAMETER    R"("M": {"type": "int", "case_sensitive": "false", "range":[0, 2147483647], "candidates":[]})"
#define PQ_BIT_SIZE_PARAMETER   R"("bit_size": {"type": "int", "case_sensitive": "false", "range":[2, 12], "candidates":[] })"
#define SQ_BIT_SIZE_PARAMETER   R"("bit_size": {"type": "string", "case_sensitive": "true", "range":[], "candidates":["4bit","6bit","8bit","8bit_uniform", "8bit_direct", "4bit_uniform", "QT_fp16"]})"

namespace Search
{

/// MyScale Valid Index Parameters
const std::string MYSCALE_VALID_INDEX_PARAMETER = "{"
        R"("SCANN": {)"     FLOAT_VECTOR_METRIC_TYPE_PARAMETER  "," VECTOR_SEARCH_ALPHA_PARAMETER "},"
        R"("FLAT": {)"      FLOAT_VECTOR_METRIC_TYPE_PARAMETER  "},"
        R"("IVFFLAT": {)"   FLOAT_VECTOR_METRIC_TYPE_PARAMETER  "," VECTOR_INDEX_IVF_BASE_PARAMETER "},"
        R"("IVFPQ": {)"     FLOAT_VECTOR_METRIC_TYPE_PARAMETER  "," VECTOR_INDEX_IVF_BASE_PARAMETER "," VECTOR_INDEX_IVF_M_PARAMETER "," PQ_BIT_SIZE_PARAMETER "},"
        R"("IVFSQ": {)"     FLOAT_VECTOR_METRIC_TYPE_PARAMETER  "," VECTOR_INDEX_IVF_BASE_PARAMETER "," SQ_BIT_SIZE_PARAMETER "},"
        R"("HNSWFLAT": {)"  FLOAT_VECTOR_METRIC_TYPE_PARAMETER  "," VECTOR_INDEX_HNSW_BASE_PARAMETER "},"
        R"("HNSWSQ": {)"    FLOAT_VECTOR_METRIC_TYPE_PARAMETER  "," VECTOR_INDEX_HNSW_BASE_PARAMETER "," SQ_BIT_SIZE_PARAMETER "},"
        R"("BINARYFLAT": {)" BINARY_VECTOR_METRIC_TYPE_PARAMETER "},"
        R"("BINARYIVF": {)"  BINARY_VECTOR_METRIC_TYPE_PARAMETER "," VECTOR_INDEX_IVF_BASE_PARAMETER "," VECTOR_INDEX_IVF_M_PARAMETER "},"
        R"("BINARYHNSW": {)" BINARY_VECTOR_METRIC_TYPE_PARAMETER "," VECTOR_INDEX_HNSW_BASE_PARAMETER "}"
    "}";

std::string getDefaultIndexType(const DataType &search_type)
{
    switch (search_type)
    {
        case DataType::FloatVector:
        {
            return "SCANN";
        }
        case DataType::BinaryVector:
        {
            return "BinaryIVF";
        }
        default:
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unsupported vector search type: %s", enumToString(search_type).c_str());
    }
}

/// The HNSW algorithm uses the corresponding HNSWFAST series algorithms by default
IndexType getVectorIndexType(std::string type, const DataType &search_type)
{
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return std::toupper(c); });

    switch (search_type)
    {
        case DataType::FloatVector:
        {
            if (type == "IVFFLAT")
                return IndexType::IVFFLAT;
            else if (type == "IVFPQ")
                return IndexType::IVFPQ;
            else if (type == "IVFSQ")
                return IndexType::IVFSQ;
            else if (type == "FLAT")
                return IndexType::FLAT;
            else if (type == "HNSWFLAT" || type == "HNSWFASTFLAT")
                return IndexType::HNSWfastFLAT;
            else if (type == "HNSWPQ" || type == "HNSWFASTPQ")
                return IndexType::HNSWPQ;
            else if (type == "HNSWSQ" || type == "HNSWFASTSQ")
                return IndexType::HNSWfastSQ;
            else if (type == "SCANN")
                return IndexType::SCANN;
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unknown index type for Float32 Vector: %s", type.c_str());
        }
        case DataType::BinaryVector:
        {
            if (type == "BINARYIVF")
                return IndexType::BinaryIVF;
            else if (type == "BINARYHNSW")
                return IndexType::BinaryHNSW;
            else if (type == "BINARYFLAT")
                return IndexType::BinaryFLAT;
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unknown index type for Binary Vector: %s", type.c_str());
        }
        default:
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unsupported vector search type: %s", enumToString(search_type).c_str());
    }
}

Metric getMetricType(std::string metric, const DataType &search_type)
{
    std::transform(metric.begin(), metric.end(), metric.begin(), [](unsigned char c) { return std::toupper(c); });

    switch (search_type)
    {
        case DataType::FloatVector:
            if (metric == "L2")
                return Metric::L2;
            else if (metric == "IP")
                return Metric::IP;
            else if (metric == "COSINE")
                return Metric::Cosine;
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unknown metirc type for Float32 Vector: %s", metric.c_str());
        case DataType::BinaryVector:
            if (metric == "HAMMING")
                return Metric::Hamming;
            else if (metric == "JACCARD")
                return Metric::Jaccard;
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unknown metirc type for Binary Vector: %s", metric.c_str());
        default:
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unsupported vector search type: %s", enumToString(search_type).c_str());
    }
}

}
