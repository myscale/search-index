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

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include <SearchIndex/Common/IndexDataIO.h>
#include <SearchIndex/Common/Utils.h>
#include <SearchIndex/SearchIndexCommon.h>

namespace Search
{

/**
 * @brief Local disk-based file store used in disk-based algorithms.
 *
 * Each Disk-based index should have exactly one file store, and it's used in
 * building, serializing & loading. Not thread-safe.
 */
template <typename IS, typename OS>
class LocalDiskFileStore : public FileStore<IS, OS>
{
public:
    // Perform read & write in 64MB blocks for maximal efficiency
    static const size_t IO_BLOCK_SIZE = 64000000;
    static inline const std::string CHECKSUM_SUFFIX = ".checksum";
    using isTerminationCall = std::function<bool()>;

    LocalDiskFileStore(
        const std::string & base_path_,
        bool use_checksum_,
        bool manage_cache_folder_ = false,
        isTerminationCall is_termination_call_ = {}) :
        base_path(base_path_),
        use_checksum(use_checksum_),
        manage_cache_folder(manage_cache_folder_),
        is_termination_call(is_termination_call_)
    {
        if (manage_cache_folder)
        {
            SI_THROW_IF_NOT_MSG(
                base_path.size() > 0 and base_path.back() == '/',
                ErrorCode::BAD_ARGUMENTS,
                "base_path should end with a slash (/) when "
                "manage_cache_folder is true");
            try
            {
                std::filesystem::create_directories(base_path);
            }
            catch (const std::exception & e)
            {
                SI_THROW_FMT(
                    ErrorCode::LOGICAL_ERROR,
                    "Failed to create cache folder %s: %s",
                    base_path.c_str(),
                    e.what());
            }
        }
    }

    ~LocalDiskFileStore() override
    {
        if (manage_cache_folder)
        {
            if (is_termination_call && is_termination_call())
            {
                SI_LOG_DEBUG("Server In terminate state, does not remove cache folder {}", base_path);
                return;
            }
            SI_LOG_INFO("Removing cache folder {}", base_path);
            if (std::filesystem::exists(base_path))
            {
                try
                {
                    std::filesystem::remove_all(base_path);
                }
                catch (const std::exception & e)
                {
                    SI_LOG_WARNING(
                        "Failed to remove cache folder {}: {}",
                        base_path,
                        e.what());
                }
            }
            else
            {
                SI_LOG_WARNING(
                    "Cache folder {} does not exist, nothing to remove",
                    base_path);
            }
        }
    }

    /// @brief Get a file path based on field name.
    std::string getFilePath(const std::string & name, bool temporary) override
    {
        auto file = base_path + name;
        name_to_files[name] = file;
        if (temporary)
        {
            temporary_files.insert(file);
        }
        return file;
    }

    int remove(const std::string & path) override
    {
        for (const auto & it : name_to_files)
        {
            if (it.second == path)
            {
                removed_files.insert(path);
                SI_LOG_INFO("Removing file {}", it.second);
                return std::remove(it.second.c_str());
            }
        }
        SI_THROW_FMT(
            ErrorCode::LOGICAL_ERROR,
            "Can't remove file not in FileStore: %s",
            path.c_str());
    }

    int removeByName(const std::string & name) override
    {
        auto it = name_to_files.find(name);
        if (it != name_to_files.end())
        {
            remove(it->second);
        }
        SI_THROW_FMT(
            ErrorCode::LOGICAL_ERROR,
            "Can't remove name not in FileStore: %s",
            name.c_str());
    }

    void removeTemporaryFiles() override
    {
        for (auto & s : temporary_files)
        {
            remove(s);
        }
    }

    /// @brief mask these files as removed (when they have been deleted elsewhere)
    void markRemoved(const std::vector<std::string> & files) override
    {
        removed_files.insert(files.begin(), files.end());
    }

    void loadFiles(
        IndexDataReader<IS> * reader,
        const std::vector<std::string> & names) override
    {
        std::vector<char> buffer(IO_BLOCK_SIZE);
        for (const auto & n : names)
        {
            auto file_name = getFilePath(n, false);
            if (use_checksum && verifyChecksum(reader, n, file_name))
            {
                SI_LOG_INFO(
                    "Checksum verification for {} and {} success, skip loading",
                    n,
                    file_name);
                // skip processing
                continue;
            }
            auto istream = reader->getFieldDataInputStream(n);
            SI_THROW_IF_NOT_FMT(
                static_cast<bool>(istream),
                ErrorCode::CANNOT_OPEN_FILE,
                "Can't open stream %s",
                n.c_str());
            std::ofstream file(file_name, std::ios::binary);
            SI_LOG_INFO("loading file {} from {}", file_name, n);

            while (*istream)
            {
                istream->read(&buffer[0], IO_BLOCK_SIZE);
                file.write(&buffer[0], istream->gcount());
                SI_THROW_IF_NOT_FMT(
                    file.good(),
                    ErrorCode::CANNOT_WRITE_TO_OSTREAM,
                    "Load file from %s to %s failed",
                    n.c_str(),
                    file_name.c_str());
            }
            file.close();

            size_t reader_checksum;
            if (use_checksum && reader->readFieldChecksum(n, reader_checksum))
            {
                // write local checksum file after loading
                writeLocalChecksumFile(file_name, reader_checksum);
            }
        }
    }

    void saveFiles(IndexDataWriter<OS> * writer) override
    {
        std::vector<char> buffer(IO_BLOCK_SIZE);
        for (const auto & it : name_to_files)
        {
            // don't save removed files
            if (removed_files.contains(it.second))
                continue;
            std::ifstream file(it.second, std::ios::binary);
            SI_THROW_IF_NOT_FMT(
                file.is_open(),
                ErrorCode::CANNOT_OPEN_FILE,
                "Opening %s for saveFiles() failed",
                it.second.c_str());
            auto ostream = writer->getFieldDataOutputStream(it.first);
            SI_LOG_INFO("saving file {} to {}", it.second, it.first);

            size_t checksum = 0;
            std::hash<std::string_view> buffer_hash;
            while (file)
            {
                file.read(&buffer[0], IO_BLOCK_SIZE);
                ostream->write(&buffer[0], file.gcount());
                if (use_checksum)
                {
                    // update the checksum value block by block
                    checksum ^= buffer_hash(
                        std::string_view(&buffer[0], file.gcount()));
                }
                SI_THROW_IF_NOT_FMT(
                    ostream->good(),
                    ErrorCode::CANNOT_WRITE_TO_OSTREAM,
                    "Save file %s to ostream %s failed",
                    it.second.c_str(),
                    it.first.c_str());
            }
            ostream->close();

            if (use_checksum)
            {
                // write to a local checksum file
                writeLocalChecksumFile(it.second, checksum);
                // save checksum to storage system
                writer->writeFieldChecksum(it.first, checksum);
            }
        }
    }

    virtual void writeLocalChecksumFile(
        const std::string & file_name, size_t checksum) override
    {
        SI_LOG_INFO(
            "Writing local checksum for file {} with checksum={}",
            file_name,
            checksum);
        std::ofstream checksum_file(
            file_name + CHECKSUM_SUFFIX, std::ios::binary);
        checksum_file.write(
            reinterpret_cast<char *>(&checksum), sizeof(size_t));
        checksum_file.close();
    }

    virtual bool removeLocalChecksumFile(const std::string & name) override
    {
        SI_LOG_INFO("Delete local checksum file {}", name);
        std::filesystem::path filePath = name + CHECKSUM_SUFFIX;
        return std::filesystem::remove(filePath);
    }

    static bool verifyChecksum(
        IndexDataReader<IS> * reader,
        const std::string & name,
        const std::string & file_name,
        std::optional<size_t> * ret_checksum = nullptr)
    {
        size_t reader_checksum;
        bool ret = reader->readFieldChecksum(name, reader_checksum);
        if (!ret)
        {
            SI_LOG_WARNING(
                "Getting checksum for {} from reader has failed", name);
            return false;
        }
        if (ret_checksum)
            *ret_checksum = reader_checksum;
        auto checksum_fname = file_name + CHECKSUM_SUFFIX;
        std::ifstream checksum_file(checksum_fname, std::ios::binary);
        if (!checksum_file.is_open())
        {
            SI_LOG_WARNING(
                "Opening checksum file {} from reader has failed",
                checksum_fname);
            return false;
        }
        if (std::filesystem::last_write_time(checksum_fname)
            < std::filesystem::last_write_time(file_name))
        {
            SI_LOG_WARNING(
                "Checksum file {} invalid, must be newer than data file",
                checksum_fname);
            return false;
        }

        size_t fs_checksum;
        checksum_file.read(
            reinterpret_cast<char *>(&fs_checksum), sizeof(size_t));
        if (reader_checksum != fs_checksum)
        {
            SI_LOG_WARNING(
                "Checksum verification failed: {} != {}",
                reader_checksum,
                fs_checksum);
            return false;
        }
        return true;
    }

private:
    std::string base_path;
    bool use_checksum;
    bool manage_cache_folder;
    std::unordered_map<std::string, std::string> name_to_files;
    std::unordered_set<std::string> removed_files;
    std::unordered_set<std::string> temporary_files;
    isTerminationCall is_termination_call;
};

}
