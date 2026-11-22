#pragma once

#ifndef RENA_CORE_TYPES_H
#define RENA_CORE_TYPES_H

#include <cstdint>
#include <string>

#include "core/common/span.h"

namespace rena::core {

    // clang-format off

    enum class lang : uint8_t {
        UNKNOWN,
        AF, AM, AR, AS, AZ, BE, BG, BN,
        BR, BS, CA, CS, CY, DA, DE, EL,
        EN, EO, ES, ET, EU, FA, FI, FR,
        FY, GA, GD, GL, GU, HA, HE, HI,
        HR, HU, HY, ID, IS, IT, JA, JV,
        KA, KK, KM, KN, KO, KU, KY, LA,
        LO, LT, LV, MG, MK, ML, MN, MR,
        MS, MY, NE, NL, NO, OM, OR, PA,
        PL, PS, PT, RO, RU, SA, SD, SI,
        SK, SL, SO, SQ, SR, SU, SV, SW,
        TA, TE, TH, TL, TR, UG, UK, UR,
        UZ, VI, XH, YI, ZH,
        UNIVERSAL
    }; // enum class lang : uint8_t

    struct category {
        int id;
        std::string name;
        std::string description;
        std::string prompt;
    }; // struct category

    struct classification_input {
        std::string text;
        std::string normalized_text;
        lang lang_hint = lang::UNKNOWN;
        common::span<int> candidate_category_ids = {};
    }; // struct classification_input

    struct classification_result {
        int category_id;
        double score;
        double raw_score;
    }; // struct classification_result

    // clang-format on

} // namespace rena::core

#endif // RENA_CORE_TYPES_H
