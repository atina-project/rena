#pragma once

#ifndef RENA_CORE_COMMON_SPAN_H
#define RENA_CORE_COMMON_SPAN_H

#include <array>
#include <cstddef>
#include <vector>

namespace rena::core::common {

    template<typename T>
    class span {
        public:
            constexpr span() noexcept = default;
            constexpr span(const T* data, std::size_t size) noexcept
                : _data(data),
                  _size(size) {}

            template<std::size_t N>
            constexpr span(const T (&arr)[N]) noexcept
                : _data(arr),
                  _size(N) {}

            template<std::size_t N>
            constexpr span(const std::array<T, N>& arr) noexcept
                : _data(arr.data()),
                  _size(N) {}

            template<std::size_t N>
            span(const std::array<T, N>&&) = delete;

            span(const std::vector<T>& vec) noexcept
                : _data(vec.data()),
                  _size(vec.size()) {}
            span(const std::vector<T>&&) = delete;

            constexpr const T* data() const noexcept { return this->_data; }
            constexpr std::size_t size() const noexcept { return this->_size; }
            constexpr bool empty() const noexcept { return this->_size == 0; }
            constexpr const T* begin() const noexcept { return this->_data; }
            constexpr const T* end() const noexcept { return this->_data + this->_size; }
            constexpr const T& operator[](std::size_t i) const noexcept { return this->_data[i]; }

            template<typename U = T>
            constexpr bool contains(const U& value) const noexcept(noexcept(*this->_data == value)) {
                for (std::size_t i = 0; i < this->_size; i++)
                {
                    if (this->_data[i] == value)
                    {
                        return true;
                    }
                }
                return false;
            }

        private:
            const T* _data = nullptr;
            std::size_t _size = 0;
    }; // class span

} // namespace rena::core::common

#endif // RENA_CORE_COMMON_SPAN_H
