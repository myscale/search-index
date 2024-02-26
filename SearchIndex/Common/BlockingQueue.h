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
#include <condition_variable>
#include <mutex>
#include <queue>

namespace Search
{

template <typename T>
class BlockingQueue
{
private:
    std::mutex mut;
    std::queue<T> private_std_queue;
    std::condition_variable cond_not_empty;
    std::condition_variable cond_not_full;
    int count{0}; // Guard with Mutex
    int max_size;

public:
    BlockingQueue(int max_size_) : max_size(max_size_) { }

    void put(T new_value)
    {
        std::unique_lock<std::mutex> lk(mut);
        //Condition takes a unique_lock and waits given the false condition
        cond_not_full.wait(lk, [this] { return count < max_size; });
        private_std_queue.push(new_value);
        count++;
        cond_not_empty.notify_one();
    }
    void take(T & value)
    {
        std::unique_lock<std::mutex> lk(mut);
        //Condition takes a unique_lock and waits given the false condition
        cond_not_empty.wait(lk, [this] { return !private_std_queue.empty(); });
        value = private_std_queue.front();
        private_std_queue.pop();
        count--;
        cond_not_full.notify_one();
    }
};

}
