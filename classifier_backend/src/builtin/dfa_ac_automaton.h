#pragma once

#ifndef RENA_CLASSIFIER_BACKEND_BUILTIN_DFA_AC_AUTOMATON_H
#define RENA_CLASSIFIER_BACKEND_BUILTIN_DFA_AC_AUTOMATON_H

#include <cstddef>
#include <cstdint>
#include <queue>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "builtin/ac_automaton_interface.h"

namespace rena::classifier_backend::builtin {

    // Tstate controls the integer width of entries in _trans (uint16_t / uint32_t / uint64_t).
    template<class Trule, class Tstate = uint32_t>
    class dfa_ac_automaton : public ac_automaton_interface<Trule> {
            static_assert(std::is_same_v<Tstate, uint16_t> || std::is_same_v<Tstate, uint32_t> ||
                              std::is_same_v<Tstate, uint64_t>,
                          "Tstate must be uint16_t, uint32_t, or uint64_t");

        public:
            using pattern_data = typename ac_automaton_interface<Trule>::pattern_data;
            using pattern_hit_callback = typename ac_automaton_interface<Trule>::pattern_hit_callback;

        public:
            dfa_ac_automaton() = default;
            ~dfa_ac_automaton() override = default;

            bool compile(const std::vector<pattern_data>& patterns, bool case_insensitive = true) override {
                this->clear();

                // Compress the 256-byte alphabet to only bytes that appear in patterns.
                // HEX ID 0 is reserved for all unseen bytes (always transitions back to root/fail).

                bool hex_used[256] = {}; // track raw byte values (0x00 ~ 0xFF) present in patterns
                for (const auto& it_pattern : patterns)
                {
                    if (it_pattern.pattern.empty())
                    {
                        continue;
                    } // empty pattern

                    for (unsigned char it_c : it_pattern.pattern)
                    {
                        if (case_insensitive && it_c >= 'A' && it_c <= 'Z')
                        {
                            it_c = static_cast<unsigned char>(it_c - 'A' + 'a');
                        } // case insensitive, use lower case
                        hex_used[it_c] = true;
                    }
                }

                uint32_t this_used_hex_id = 1;
                for (int i = 0; i < 256; i++)
                {
                    if (hex_used[i])
                    {
                        if (this_used_hex_id > 255)
                        {
                            this->clear();
                            return false;
                        } // class 0 is reserved, so at most 255 distinct bytes fit in uint8_t
                        this->_byte_map[i] = static_cast<uint8_t>(this_used_hex_id);
                        this_used_hex_id++;
                    }
                }
                if (case_insensitive)
                {
                    for (int i = 'A'; i <= 'Z'; i++)
                    {
                        this->_byte_map[i] = this->_byte_map[i - 'A' + 'a'];
                    }
                } // case insensitive, map all upper cases to its lower case

                this->_alphabet_size = this_used_hex_id;
                if (this->_alphabet_size == 1)
                {
                    return false;
                } // empty alphabet

                // Build a temp trie over compressed byte-class IDs.

                // clang-format off
                struct node {
                    std::vector<std::pair<uint8_t, std::size_t>> children; // (compressed byte class ID, child node index in trie)
                    std::vector<hit_entry> own_hits;                       // rules terminating directly at this trie node
                }; // struct node
                // clang-format on

                std::vector<node> trie(1); // node 0 is the root
                for (const auto& it_pattern : patterns)
                {
                    if (it_pattern.pattern.empty())
                    {
                        continue;
                    } // empty pattern

                    std::size_t current_node = 0; // start from root
                    for (unsigned char it_raw_c : it_pattern.pattern)
                    {
                        const uint8_t hex_id = this->_byte_map[it_raw_c];
                        std::size_t next_node = 0;
                        for (const auto& [it_edge_hex_id, it_child_id] : trie[current_node].children)
                        {
                            if (it_edge_hex_id == hex_id)
                            {
                                next_node = it_child_id;
                                break;
                            } // existing child edge found for this byte class
                        }
                        if (next_node == 0)
                        {
                            next_node = trie.size();
                            trie.emplace_back();
                            trie[current_node].children.emplace_back(hex_id, next_node);
                        } // no edge for this byte class yet; append a new child node
                        current_node = next_node;
                    }

                    hit_entry hit = { it_pattern.hit_rule, static_cast<uint32_t>(it_pattern.pattern.size()) };
                    trie[current_node].own_hits.push_back(hit);
                }

                const std::size_t n_nodes = trie.size();
                if (static_cast<uint64_t>(n_nodes) >= static_cast<uint64_t>(HIT_BIT))
                {
                    this->clear();
                    return false;
                } // MSB of Tstate is reserved for HIT_BIT, so state count must be < 2^(bits-1)

                // Reorder node IDs in BFS order.
                // This keeps shallow (frequently visited) states contiguous at the front of _trans for cache locality,
                // and guarantees fail[state] < state for all non-root states in the linear passes below.

                std::vector<Tstate> bfs_order;               // new_node_id -> old_node_id
                std::vector<Tstate> node_id_map(n_nodes, 0); // old_node_id -> new_node_id
                bfs_order.reserve(n_nodes);

                std::queue<Tstate> q;
                q.push(0);
                bfs_order.push_back(0);
                node_id_map[0] = 0;
                // root always keeps ID 0

                while (!q.empty())
                {
                    const Tstate old_node_id = q.front();
                    q.pop();
                    for (const auto& [_, it_old_child_id] : trie[old_node_id].children)
                    {
                        const Tstate new_child_id = static_cast<Tstate>(bfs_order.size());
                        const Tstate old_child_id = static_cast<Tstate>(it_old_child_id);
                        node_id_map[old_child_id] = new_child_id;
                        bfs_order.push_back(old_child_id);
                        q.push(old_child_id);
                    }
                }

                // Build full DFA transition table (_trans) and compute failure links (fail).
                // Missing transitions are filled with fail-state transitions so match() needs no runtime fail loop.

                this->_trans.assign(n_nodes * this->_alphabet_size, 0);
                std::vector<Tstate> fail(n_nodes, 0);

                for (std::size_t new_node_id = 0; new_node_id < n_nodes; new_node_id++)
                {
                    const Tstate old_node_id = bfs_order[new_node_id];
                    Tstate* row = this->_trans.data() + new_node_id * this->_alphabet_size;
                    for (const auto& [it_hex_id, it_old_child_id] : trie[old_node_id].children)
                    {
                        row[it_hex_id] = node_id_map[it_old_child_id];
                    }
                } // populate direct trie transitions using the BFS-reordered state IDs
                // Root (node 0) is already complete: unseen edges stay 0 (self-loop) and depth-1 children have fail =
                // 0. For node_id >= 1, BFS ordering guarantees fail[node_id] < node_id, so fail_row is already
                // finalized.

                for (std::size_t node_id = 1; node_id < n_nodes; node_id++)
                {
                    Tstate* row = this->_trans.data() + node_id * this->_alphabet_size;
                    const Tstate* fail_row =
                        this->_trans.data() + static_cast<std::size_t>(fail[node_id]) * this->_alphabet_size;
                    for (uint32_t i = 0; i < this->_alphabet_size; i++)
                    {
                        if (row[i] != 0)
                        {
                            fail[row[i]] = fail_row[i]; // direct child: set its failure state
                        }
                        else
                        {
                            row[i] = fail_row[i]; // missing edge: fall back directly to fail state's transition
                        }
                    }
                }

                // Flatten hits per state (own hits + inherited suffix hits via fail links)
                // into a contiguous _hits pool so match() never chases fail links at runtime.

                this->_hit_slices.resize(n_nodes);
                for (std::size_t node_id = 0; node_id < n_nodes; node_id++)
                {
                    const uint32_t offset = static_cast<uint32_t>(this->_hits.size());
                    const auto& own = trie[bfs_order[node_id]].own_hits;
                    this->_hits.insert(this->_hits.end(), own.begin(), own.end());

                    if (node_id != 0)
                    {
                        // fail[node_id] < node_id in BFS order, so its flattened slice is already populated.
                        const auto fail_slice = this->_hit_slices[fail[node_id]];
                        if (fail_slice.count > 0)
                        {
                            this->_hits.reserve(this->_hits.size() + fail_slice.count);
                            // pre-reserve to avoid reallocation while copying from earlier entries in _hits
                            for (uint32_t i = 0; i < fail_slice.count; i++)
                            {
                                this->_hits.push_back(this->_hits[fail_slice.offset + i]);
                            }
                        }
                    }

                    const uint32_t count = static_cast<uint32_t>(this->_hits.size()) - offset;
                    this->_hit_slices[node_id] = { offset, count };
                }

                // Bake HIT_BIT into transition entries whose target state produces at least one hit.
                // This lets match() skip accessing _hit_slices on non-matching bytes (hot/cold split).
                for (Tstate& it_node : this->_trans)
                {
                    if (this->_hit_slices[it_node].count > 0)
                    {
                        it_node |= HIT_BIT;
                    }
                }

                return true;
            }

            void match(std::string_view text, pattern_hit_callback hit_cb, void* user_data = nullptr) const override {
                if (this->_trans.empty() || !hit_cb)
                {
                    return;
                } // automaton not compiled or null callback

                const auto* text_data = reinterpret_cast<const uint8_t*>(text.data());
                const std::size_t text_len = text.size();
                const Tstate* trans = this->_trans.data();
                const std::size_t alphabet_size = this->_alphabet_size;

                Tstate current_node = 0;
                for (std::size_t i = 0; i < text_len; i++)
                {
                    // Hot path: map raw byte to compressed class ID and perform a single DFA table lookup.
                    const uint8_t hex_id = this->_byte_map[text_data[i]];
                    current_node = trans[static_cast<std::size_t>(current_node & STATE_MASK) * alphabet_size + hex_id];

                    // Cold path: only access _hit_slices and _hits when the destination state has hits.
                    if ((current_node & HIT_BIT) != 0)
                    {
                        const auto& slice = this->_hit_slices[current_node & STATE_MASK];
                        for (uint32_t j = 0; j < slice.count; j++)
                        {
                            const auto& it_hit = this->_hits[slice.offset + j];
                            // Half-open byte range [begin_pos, end_pos) in text
                            const std::size_t begin_pos = i + 1 - it_hit.pattern_len;
                            const std::size_t end_pos = i + 1;
                            hit_cb(it_hit.rule, begin_pos, end_pos, user_data);
                        }
                    }
                }
                return;
            }

            void clear() noexcept override {
                for (uint8_t& it_b : this->_byte_map)
                {
                    it_b = 0;
                }
                this->_alphabet_size = 1;
                this->_trans.clear();
                this->_hit_slices.clear();
                this->_hits.clear();
                return;
            }

        private:
            // MSB of each _trans entry marks whether the target state has hits; remaining bits store state ID.
            static constexpr Tstate HIT_BIT = static_cast<Tstate>(Tstate(1) << (sizeof(Tstate) * 8 - 1));
            static constexpr Tstate STATE_MASK = static_cast<Tstate>(~HIT_BIT);

            // clang-format off

            struct hit_slice {
                uint32_t offset = 0; // start index in _hits
                uint32_t count = 0;  // number of flattened hits for this state
            }; // struct hit_slice

            struct hit_entry {
                Trule rule;
                uint32_t pattern_len; // matched pattern length in bytes
            }; // struct hit_entry

            // clang-format on

            // Hot data: accessed on every input byte during match()
            uint8_t _byte_map[256] = {}; // raw byte (0x00~0xFF) -> compressed class ID (0 = unseen)
            uint32_t _alphabet_size = 1; // number of compressed classes (including class 0)
            std::vector<Tstate> _trans;  // flattened DFA table [n_nodes * _alphabet_size] in BFS state order

            // Cold data: accessed only when HIT_BIT is set
            std::vector<hit_slice> _hit_slices; // per-state slice into _hits, indexed by BFS state ID
            std::vector<hit_entry> _hits;       // contiguous pool of all flattened pattern hits

    }; // class dfa_ac_automaton

} // namespace rena::classifier_backend::builtin

#endif // RENA_CLASSIFIER_BACKEND_BUILTIN_DFA_AC_AUTOMATON_H
