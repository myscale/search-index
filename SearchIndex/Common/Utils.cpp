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

#include "Utils.h"
#include <fstream>
#include <sys/resource.h>

namespace Search
{

std::string getRSSUsage()
{
    std::string line;

    // Create the file path for the process status
    std::string path = "/proc/self/status";

    // Open the file for reading
    std::ifstream file(path);
    if (file.is_open())
    {
        // Search for the line containing the RSS
        while (std::getline(file, line))
        {
            if (line.substr(0, 5) == "VmRSS")
            {
                return line;
            }
        }
        file.close();
    }
    else
    {
        SI_LOG_ERROR("Error opening file.");
        return "";
    }
    return "";
}

void printMemoryUsage(const std::string & header)
{
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    SI_LOG_INFO(
        "{} Current memory usage: {}, peak memory usage: {} MB",
        header,
        getRSSUsage(),
        usage.ru_maxrss / 1024);
}

}