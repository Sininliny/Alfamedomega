#include "alfa/dsp/BudgetGuard.h"

#include "alfa/dsp/Voice.h"

#include <algorithm>
#include <cmath>

namespace alfa::dsp
{

void BudgetGuard::prepare(double sampleRate, Tier tier) noexcept
{
    sampleRate_ = sampleRate;
    tier_ = std::clamp(static_cast<int>(tier), 0, 3);
    budget_ = budgetGuardFor(tier).maxShare;
    reset();
}

void BudgetGuard::setCostTable(const CostTable* table) noexcept
{
    table_ = table != nullptr ? table : &generated::kCostTable;
}

void BudgetGuard::setTrigger(GuardTrigger trigger) noexcept
{
    trigger_ = trigger;
    reset();
}

void BudgetGuard::reset() noexcept
{
    levels_.fill(0);
    families_.fill(Family::Off);
    orders_.fill(0);
    measured_ = false;
    loadAvg_ = 0.0f;
    modelAvg_ = 0.0f;
    lastModel_ = 0.0f;
}

void BudgetGuard::observe(double seconds, int samples) noexcept
{
    if (samples <= 0 || !(seconds >= 0.0) || sampleRate_ <= 0.0)
        return;
    const double duration = samples / sampleRate_;
    const auto load = static_cast<float>(seconds / duration);
    if (!measured_)
    {
        // Start the average at the model's prediction rather than at the first (cold) block.
        loadAvg_ = lastModel_;
        modelAvg_ = lastModel_;
        measured_ = true;
    }
    const auto alpha = static_cast<float>(1.0 - std::exp(-duration / kLoadTau));
    loadAvg_ += alpha * (load - loadAvg_);
    modelAvg_ += alpha * (lastModel_ - modelAvg_);
}

float BudgetGuard::share(const float (&ns)[2]) const noexcept
{
    // Linear in the sample rate between the two measured rates (hop-rate work, such as the
    // OSC table rebuild, does not double with the rate; per-sample work does).
    const double t = (sampleRate_ - 48000.0) / 48000.0;
    const double perSample =
        std::max(0.0, static_cast<double>(ns[0]) + t * static_cast<double>(ns[1] - ns[0]));
    const auto fit = [](float f)
    {
        return f > 0.0f ? f : 1.0f;
    };
    const float r0 = fit(table_->rateScale[0]), r1 = fit(table_->rateScale[1]);
    const float rate = r0 + static_cast<float>(std::clamp(t, 0.0, 1.0)) * (r1 - r0);
    return static_cast<float>(perSample * sampleRate_ * 1.0e-9) * rate * scale_ * kSafetyMargin;
}

float BudgetGuard::primitiveShare(int primitive) const noexcept
{
    return share(table_->primitive[tier_][primitive]);
}

float BudgetGuard::voiceSlotShare(Family f, int quality) const noexcept
{
    return share(table_->voiceSlot[tier_][static_cast<int>(f)][std::clamp(quality, 0, 1)]);
}

float BudgetGuard::instanceSlotShare(Family f) const noexcept
{
    return share(table_->instanceSlot[tier_][static_cast<int>(f)][0]);
}

GuardReport BudgetGuard::run(Voice* const* voices, const std::uint64_t* startOrder, int voiceCount,
                             const Family* instanceFamilies, int instanceSlots) noexcept
{
    GuardReport report;
    report.budgetShare = budget_;
    const auto& weights = budgetGuardFor(static_cast<Tier>(tier_)).familyWeightPct;
    const bool persist = trigger_ == GuardTrigger::Measured && enabled_;

    float total = share(table_->instanceBase);
    for (int k = 0; k < instanceSlots; ++k)
        if (instanceFamilies[k] != Family::Off)
            total += instanceSlotShare(instanceFamilies[k]);

    int count = 0;
    for (int v = 0; v < std::min(voiceCount, kMaxVoices); ++v)
    {
        Voice* voice = voices[v];
        const int slots = voice->slotCount();
        const auto base = static_cast<std::size_t>(v) * kMaxVoiceSlots;
        const bool newNote = orders_[static_cast<std::size_t>(v)] != startOrder[v];
        orders_[static_cast<std::size_t>(v)] = startOrder[v];
        if (!voice->active())
        {
            for (int k = 0; k < slots; ++k)
            {
                voice->setSlotGuard(k, 0);
                levels_[base + static_cast<std::size_t>(k)] = 0;
            }
            continue;
        }
        total += share(table_->voiceBase[tier_]);
        for (int p = 0; p < kPrimitiveCount; ++p)
            if (voice->primitiveActive(p))
                total += primitiveShare(p);
        for (int k = 0; k < slots; ++k)
        {
            const auto idx = base + static_cast<std::size_t>(k);
            const Family f = voice->requestedFamily(k);
            int level = persist ? levels_[idx] : 0;
            if (newNote || f != families_[idx])
                level = 0;
            families_[idx] = f;
            levels_[idx] = 0;
            if (f == Family::Off || count >= static_cast<int>(candidates_.size()))
            {
                voice->setSlotGuard(k, 0);
                if (f != Family::Off)
                    total += voiceSlotShare(f, 0);
                continue;
            }
            const float full = voiceSlotShare(f, 0);
            const float reduced = voiceSlotShare(f, 1);
            // A reduced mode that saves under 5% is not worth its audible change.
            Candidate c{voice,
                        startOrder[v],
                        {full, reduced < 0.95f * full ? reduced : full, 0.0f},
                        weights[static_cast<int>(f)],
                        k,
                        level,
                        idx};
            if (c.level == 1 && c.cost[1] >= c.cost[0])
                c.level = 0;
            total += c.cost[c.level];
            candidates_[static_cast<std::size_t>(count++)] = c;
        }
    }

    // Measured: the averaged load, plus whatever the model says changed since (new notes, new
    // families, the guard's own decisions), so bursts are caught before the average moves.
    const bool useLoad = trigger_ == GuardTrigger::Measured && measured_;
    const float offset = useLoad ? loadAvg_ - modelAvg_ : 0.0f;
    const auto predicted = [&]
    {
        return std::max(0.0f, total + offset);
    };
    report.estimatedShare = predicted();

    if (enabled_)
    {
        bool degraded = false;
        while (predicted() > budget_)
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
            degraded = true;
        }

        // Measured: one step back per block while the load stays under the restore threshold
        // after the step. The lightest step goes first (ties: highest weight, oldest note).
        const float restoreBelow = kRestoreFraction * budget_;
        if (persist && !degraded && predicted() < restoreBelow)
        {
            Candidate* pick = nullptr;
            int pickLevel = 0;
            float pickDelta = 0.0f;
            for (int i = 0; i < count; ++i)
            {
                Candidate& c = candidates_[static_cast<std::size_t>(i)];
                if (c.level == 0)
                    continue;
                const int prev = c.level == 2 && c.cost[1] < c.cost[0] ? 1 : 0;
                const float delta = c.cost[prev] - c.cost[c.level];
                if (pick == nullptr || delta < pickDelta ||
                    (delta == pickDelta &&
                     (c.weight > pick->weight || (c.weight == pick->weight && c.order < pick->order))))
                {
                    pick = &c;
                    pickLevel = prev;
                    pickDelta = delta;
                }
            }
            if (pick != nullptr && predicted() + pickDelta <= restoreBelow)
            {
                total += pickDelta;
                pick->level = pickLevel;
            }
        }
    }

    for (int i = 0; i < count; ++i)
    {
        const Candidate& c = candidates_[static_cast<std::size_t>(i)];
        c.voice->setSlotGuard(c.slot, c.level);
        levels_[c.index] = static_cast<std::int8_t>(c.level);
        report.downgraded += c.level == 1 ? 1 : 0;
        report.bypassed += c.level == 2 ? 1 : 0;
    }
    lastModel_ = total;
    report.finalShare = predicted();
    report.measuredShare = measured_ ? loadAvg_ : 0.0f;
    return report;
}

} // namespace alfa::dsp
