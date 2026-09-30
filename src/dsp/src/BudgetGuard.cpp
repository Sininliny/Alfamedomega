#include "alfa/dsp/BudgetGuard.h"

#include "alfa/dsp/Voice.h"

#include <algorithm>

namespace alfa::dsp
{

void BudgetGuard::prepare(double sampleRate, Tier tier) noexcept
{
    sampleRate_ = sampleRate;
    tier_ = std::clamp(static_cast<int>(tier), 0, 3);
    budget_ = budgetGuardFor(tier).maxShare;
}

float BudgetGuard::share(const float (&ns)[2]) const noexcept
{
    // Linear in the sample rate between the two measured rates (hop-rate work, such as the
    // OSC table rebuild, does not double with the rate; per-sample work does).
    const double t = (sampleRate_ - 48000.0) / 48000.0;
    const double perSample =
        std::max(0.0, static_cast<double>(ns[0]) + t * static_cast<double>(ns[1] - ns[0]));
    return static_cast<float>(perSample * sampleRate_ * 1.0e-9) * scale_ * kSafetyMargin;
}

float BudgetGuard::primitiveShare(int primitive) const noexcept
{
    return share(generated::kCostTable.primitive[tier_][primitive]);
}

float BudgetGuard::voiceSlotShare(Family f, int quality) const noexcept
{
    return share(generated::kCostTable.voiceSlot[tier_][static_cast<int>(f)][std::clamp(quality, 0, 1)]);
}

float BudgetGuard::instanceSlotShare(Family f) const noexcept
{
    return share(generated::kCostTable.instanceSlot[tier_][static_cast<int>(f)][0]);
}

GuardReport BudgetGuard::run(Voice* const* voices, const std::uint64_t* startOrder, int voiceCount,
                             const Family* instanceFamilies, int instanceSlots) noexcept
{
    GuardReport report;
    report.budgetShare = budget_;
    const auto& weights = budgetGuardFor(static_cast<Tier>(tier_)).familyWeightPct;

    float total = share(generated::kCostTable.instanceBase);
    for (int k = 0; k < instanceSlots; ++k)
        if (instanceFamilies[k] != Family::Off)
            total += instanceSlotShare(instanceFamilies[k]);

    int count = 0;
    for (int v = 0; v < voiceCount; ++v)
    {
        Voice* voice = voices[v];
        const int slots = voice->slotCount();
        if (!voice->active())
        {
            for (int k = 0; k < slots; ++k)
                voice->setSlotGuard(k, 0);
            continue;
        }
        total += share(generated::kCostTable.voiceBase[tier_]);
        for (int p = 0; p < kPrimitiveCount; ++p)
            if (voice->primitiveActive(p))
                total += primitiveShare(p);
        for (int k = 0; k < slots; ++k)
        {
            voice->setSlotGuard(k, 0);
            const Family f = voice->requestedFamily(k);
            if (f == Family::Off)
                continue;
            const float full = voiceSlotShare(f, 0);
            total += full;
            if (count < static_cast<int>(candidates_.size()))
            {
                const float reduced = voiceSlotShare(f, 1);
                // A reduced mode that saves under 5% is not worth its audible change.
                candidates_[static_cast<std::size_t>(count++)] = {
                    voice,
                    startOrder[v],
                    {full, reduced < 0.95f * full ? reduced : full, 0.0f},
                    weights[static_cast<int>(f)],
                    k,
                    0};
            }
        }
    }
    report.estimatedShare = total;

    if (enabled_)
    {
        while (total > budget_)
        {
            // Heaviest remaining slot first; ties: lowest family weight, then the newest note.
            Candidate* victim = nullptr;
            for (int i = 0; i < count; ++i)
            {
                Candidate& c = candidates_[static_cast<std::size_t>(i)];
                if (c.level >= 2)
                    continue;
                if (victim == nullptr)
                {
                    victim = &c;
                    continue;
                }
                const float a = c.cost[c.level], b = victim->cost[victim->level];
                if (a > b || (a == b && (c.weight < victim->weight ||
                                         (c.weight == victim->weight && c.order > victim->order))))
                    victim = &c;
            }
            if (victim == nullptr)
                break; // every voice slot is bypassed; the rest cannot be degraded
            const int next = victim->level == 0 && victim->cost[1] < victim->cost[0] ? 1 : 2;
            total -= victim->cost[victim->level] - victim->cost[next];
            victim->level = next;
        }
        for (int i = 0; i < count; ++i)
        {
            const Candidate& c = candidates_[static_cast<std::size_t>(i)];
            c.voice->setSlotGuard(c.slot, c.level);
            report.downgraded += c.level == 1 ? 1 : 0;
            report.bypassed += c.level == 2 ? 1 : 0;
        }
    }
    report.finalShare = total;
    return report;
}

} // namespace alfa::dsp
