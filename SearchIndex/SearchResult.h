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
#include <cstdlib>
#include <span>
#include <unordered_set>
#include <SearchIndex/SearchIndexCommon.h>

namespace Search
{

struct ResultEntry
{
    idx_t label;
    float distance;
    int res_ind;
};

struct SearchResult
{
    explicit SearchResult(
        idx_t nq_,
        size_t * lims_,
        idx_t * labels_,
        float * distances_,
        bool own_data_ = true) :
        nq(nq_),
        lims(lims_),
        labels(labels_),
        distances(distances_),
        own_data(own_data_)
    {
    }

    // To be consistent with previous API,
    // labels/distances might contain -1 as null entries
    static std::shared_ptr<SearchResult> createTopKHolder(idx_t nq, size_t topK)
    {
        size_t * lims = new size_t[nq + 1];
        idx_t * labels = new idx_t[nq * topK];
        float * distances = new float[nq * topK];

        for (idx_t i = 0; i <= nq; i++)
            lims[i] = i * topK;
        std::fill(labels, labels + nq * topK, -1);
        std::fill(distances, distances + nq * topK, -1);
        auto res
            = std::make_shared<SearchResult>(nq, lims, labels, distances, true);
        res->num_candidates = static_cast<int>(topK);
        return res;
    }

    // return -1 if invalid
    int getNumCandidates() const { return num_candidates; }

    idx_t * getResultIndices() { return labels; }

    float * getResultDistances() { return distances; }

    size_t numQueries() { return nq; }

    size_t getResultLength(size_t k)
    {
        SI_THROW_IF_NOT(k < nq, ErrorCode::LOGICAL_ERROR);
        return lims[k + 1] - lims[k];
    }

    std::span<idx_t> getResultIndices(size_t k)
    {
        return std::span(labels + lims[k], getResultLength(k));
    }

    std::span<float> getResultDistances(size_t k)
    {
        return std::span(distances + lims[k], getResultLength(k));
    }

    virtual ~SearchResult()
    {
        if (own_data)
        {
            delete[] lims;
            delete[] labels;
            delete[] distances;
        }
    }

    // merge the top candidates from multiple SearchResult
    // if truncate_input_results = true, will set -1 to the rest of the input
    static std::shared_ptr<SearchResult> merge(
        std::vector<std::shared_ptr<SearchResult>> & results,
        Metric metric,
        size_t num_candidates,
        bool truncate_input_results = false)
    {
        SI_THROW_IF_NOT(!results.empty(), ErrorCode::BAD_ARGUMENTS);
        size_t nq = results[0]->numQueries();
        auto cmp = metric == Metric::IP
            ? [](const ResultEntry & a, const ResultEntry & b)
        { return a.distance > b.distance; }
            : [](const ResultEntry & a, const ResultEntry & b)
        { return a.distance < b.distance; };
        auto merged = SearchResult::createTopKHolder(nq, num_candidates);

        for (size_t q_ind = 0; q_ind < nq; ++q_ind)
        {
            // combine the top results from each input
            int r_ind = 0;
            std::vector<ResultEntry> entries;
            for (auto res : results)
            {
                auto indices = res->getResultIndices(q_ind);
                auto dists = res->getResultDistances(q_ind);
                for (size_t i = 0; i < res->getResultLength(q_ind); ++i)
                {
                    // skip invalid entries
                    if (indices[i] < 0)
                        continue;
                    entries.emplace_back(
                        ResultEntry{indices[i], dists[i], r_ind});
                }
                r_ind++;
            }
            // sort and compute the merged_results
            std::sort(entries.begin(), entries.end(), cmp);
            std::vector<std::unordered_set<idx_t>> valid_entries(
                results.size());
            for (size_t i = 0; i < num_candidates && i < entries.size(); ++i)
            {
                merged->labels[q_ind * num_candidates + i] = entries[i].label;
                merged->distances[q_ind * num_candidates + i]
                    = entries[i].distance;
                if (truncate_input_results)
                    valid_entries[entries[i].res_ind].insert(entries[i].label);
            }
            // truncate the input results
            if (truncate_input_results)
            {
                r_ind = 0;
                for (auto res : results)
                {
                    auto indices = res->getResultIndices(q_ind);
                    for (size_t i = 0; i < res->getResultLength(q_ind); ++i)
                    {
                        if (!valid_entries[r_ind].count(indices[i]))
                            indices[i] = -1;
                    }
                    r_ind++;
                }
            }
        }
        return merged;
    }

private:
    /// number of results for each query (only valid for topK search)
    int num_candidates{-1};
    size_t nq; // nb of queries
    size_t * lims; // size (nq + 1)
    idx_t * labels; // result for query i is labels[lims[i]:lims[i+1]]
    float * distances; // corresponding distances (not sorted)
    bool own_data;
};
}
