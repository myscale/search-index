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

#include <SearchIndex/VectorIndex.h>

namespace Search
{

template <typename IS, typename OS, IDSelector IDS, DataType dataType>
int VectorIndex<IS, OS, IDS, dataType>::computeFirstStageNumCandidates(
    IndexType index_type,
    bool disk_mode,
    int64_t num_data,
    int64_t data_dim,
    int32_t topK,
    Parameters params)
{
    SI_THROW_FMT(
        ErrorCode::BAD_ARGUMENTS,
        "Unsupported index type for computing num_candidates: %s",
        enumToString(index_type).c_str());
}

// instantiate the template class
template class VectorIndex<
    AbstractIStream,
    AbstractOStream,
    DenseBitmap,
    DataType::FloatVector>;

}
