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

#include "DenseDataLayer.h"
#include <faiss/utils/distances.h>

namespace Search
{

void ComputeDistanceSubset(
    DenseDataLayer<float> * l,
    Metric metric,
    const float * x, // query vector data
    idx_t n, // number of query data points in x
    const std::span<idx_t> labels, // labels of the subset to compute distance
    float * distances) // output distances, size n * labels.size()
{
    // use L2sqr for L2/Cosine distance
    auto dist_func = metric == Metric::IP ? faiss::fvec_inner_products_by_idx
                                          : faiss::fvec_L2sqr_by_idx;

    size_t batch_size = labels.size();
    for (size_t st = 0, j = 0; st < labels.size(); st = j)
    {
        // compute current batch, skip -1 labels
        std::vector<idx_t> batch_idx;
        std::vector<idx_t> batch_labels;
        for (j = st; j < labels.size() && batch_labels.size() < batch_size; ++j)
        {
            if (labels[j] != -1)
            {
                batch_idx.push_back(j);
                batch_labels.push_back(labels[j]);
            }
        }

        const float * batch_data = nullptr;
        int64_t * batch_labels_ptr = nullptr;

        std::vector<float> batch_data_vec;
        std::vector<int64_t> batch_labels_vec;
        batch_data = l->getDataPtr(0);
        batch_labels_ptr = batch_labels.data();
        std::vector<float> batch_dist;
        batch_dist.resize(n * batch_labels.size());

        // compute distance between x and batch_data
        dist_func(
            batch_dist.data(),
            x,
            batch_data,
            batch_labels_ptr,
            l->dataDimension(),
            n,
            batch_labels.size());

        // copy computed dist matrix to output distances
        for (size_t ix = 0; ix < n; ++ix)
            for (size_t k = 0; k < batch_idx.size(); ++k)
                distances[ix * labels.size() + batch_idx[k]]
                    = batch_dist[ix * batch_idx.size() + k];
    }
}

}
