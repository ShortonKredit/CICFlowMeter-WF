# Feature Definitions: Website Fingerprinting Metadata Features (74 Features)

This document provides mathematical formulas, group classifications, references to original Java `CICFlowMeter` logic, and notes on edge cases/approximations for the 74-feature metadata feature bank.

---

## 1. Group A: 58 CICFlowMeter-style features

These features statistical properties of packet durations, bytes, packet counts, inter-arrival times (IAT), active/idle cycles, subflows, and bulk metrics, adapted for Website Fingerprinting (WF) packet trace schemas.

| # | Feature Name | Group | Formula / Definition | Reference / Port Notes |
|---|---|---|---|---|
| 1 | `flow_duration` | CIC | $t_{last} - t_{first}$ | Relative flow duration in seconds. If $< 0$, defaults to 0. |
| 2 | `flow_bytes_per_s` | CIC | $\frac{\sum \text{length}}{\text{flow\_duration}}$ | Returns 0 if $\text{flow\_duration} = 0$. |
| 3 | `flow_packets_per_s` | CIC | $\frac{N_{total}}{\text{flow\_duration}}$ | Returns 0 if $\text{flow\_duration} = 0$. |
| 4 | `total_fwd_packets` | CIC | $\sum_{i} [\text{direction}_i == 1]$ | Count of outgoing packets. |
| 5 | `total_bwd_packets` | CIC | $\sum_{i} [\text{direction}_i == -1]$ | Count of incoming packets. |
| 6 | `total_len_fwd_packets` | CIC | $\sum \text{length}_{fwd}$ | Total outgoing bytes. |
| 7 | `total_len_bwd_packets` | CIC | $\sum \text{length}_{bwd}$ | Total incoming bytes. |
| 8 | `fwd_pkt_len_min` | CIC | $\min(\text{length}_{fwd})$ | Returns 0 if $N_{fwd} == 0$. |
| 9 | `fwd_pkt_len_max` | CIC | $\max(\text{length}_{fwd})$ | Returns 0 if $N_{fwd} == 0$. |
| 10 | `fwd_pkt_len_mean` | CIC | $\text{mean}(\text{length}_{fwd})$ | Returns 0 if $N_{fwd} == 0$. |
| 11 | `fwd_pkt_len_std` | CIC | $\text{std}(\text{length}_{fwd})$ | Sample std (ddof=1). Returns 0 if $N_{fwd} < 2$. |
| 12 | `bwd_pkt_len_min` | CIC | $\min(\text{length}_{bwd})$ | Returns 0 if $N_{bwd} == 0$. |
| 13 | `bwd_pkt_len_max` | CIC | $\max(\text{length}_{bwd})$ | Returns 0 if $N_{bwd} == 0$. |
| 14 | `bwd_pkt_len_mean` | CIC | $\text{mean}(\text{length}_{bwd})$ | Returns 0 if $N_{bwd} == 0$. |
| 15 | `bwd_pkt_len_std` | CIC | $\text{std}(\text{length}_{bwd})$ | Sample std (ddof=1). Returns 0 if $N_{bwd} < 2$. |
| 16 | `flow_iat_mean` | CIC | $\text{mean}(\text{IAT}_{flow})$ | $\text{IAT}_i = \max(0.0, t_i - t_{i-1})$. Clamped to be non-negative. |
| 17 | `flow_iat_std` | CIC | $\text{std}(\text{IAT}_{flow})$ | Sample std (ddof=1). Clamped to be non-negative. |
| 18 | `flow_iat_max` | CIC | $\max(\text{IAT}_{flow})$ | Clamped to be non-negative. |
| 19 | `flow_iat_min` | CIC | $\min(\text{IAT}_{flow})$ | Clamped to be non-negative. |
| 20 | `fwd_iat_min` | CIC | $\min(\text{IAT}_{fwd})$ | Clamped to be non-negative. |
| 21 | `fwd_iat_max` | CIC | $\max(\text{IAT}_{fwd})$ | Clamped to be non-negative. |
| 22 | `fwd_iat_mean` | CIC | $\text{mean}(\text{IAT}_{fwd})$ | Clamped to be non-negative. |
| 23 | `fwd_iat_std` | CIC | $\text{std}(\text{IAT}_{fwd})$ | Sample std (ddof=1). Clamped to be non-negative. |
| 24 | `fwd_iat_total` | CIC | $\sum \text{IAT}_{fwd}$ | Equivalent to $t_{fwd\_last} - t_{fwd\_first}$. |
| 25 | `bwd_iat_min` | CIC | $\min(\text{IAT}_{bwd})$ | Clamped to be non-negative. |
| 26 | `bwd_iat_max` | CIC | $\max(\text{IAT}_{bwd})$ | Clamped to be non-negative. |
| 27 | `bwd_iat_mean` | CIC | $\text{mean}(\text{IAT}_{bwd})$ | Clamped to be non-negative. |
| 28 | `bwd_iat_std` | CIC | $\text{std}(\text{IAT}_{bwd})$ | Sample std (ddof=1). Clamped to be non-negative. |
| 29 | `bwd_iat_total` | CIC | $\sum \text{IAT}_{bwd}$ | Equivalent to $t_{bwd\_last} - t_{bwd\_first}$. |
| 30 | `fwd_packets_per_s` | CIC | $\frac{N_{fwd}}{\text{flow\_duration}}$ | Outgoing packet rate. |
| 31 | `bwd_packets_per_s` | CIC | $\frac{N_{bwd}}{\text{flow\_duration}}$ | Incoming packet rate. |
| 32 | `pkt_len_min` | CIC | $\min(\text{length}_{flow})$ | Minimum packet size. |
| 33 | `pkt_len_max` | CIC | $\max(\text{length}_{flow})$ | Maximum packet size. |
| 34 | `pkt_len_mean` | CIC | $\text{mean}(\text{length}_{flow})$ | Mean packet size. |
| 35 | `pkt_len_std` | CIC | $\text{std}(\text{length}_{flow})$ | Sample std (ddof=1). |
| 36 | `pkt_len_var` | CIC | $\text{var}(\text{length}_{flow})$ | Sample variance (ddof=1). |
| 37 | `down_up_ratio` | CIC | $\frac{N_{bwd}}{N_{fwd}}$ | Returns 0 if $N_{fwd} == 0$. |
| 38 | `avg_packet_size` | CIC | $\text{mean}(\text{length}_{flow})$ | Same as `pkt_len_mean`. (Intentional duplicate for CIC compatibility). |
| 39 | `fwd_segment_size_avg`| CIC | $\text{mean}(\text{length}_{fwd})$ | Same as `fwd_pkt_len_mean`. (Intentional duplicate for CIC compatibility). |
| 40 | `bwd_segment_size_avg`| CIC | $\text{mean}(\text{length}_{bwd})$ | Same as `bwd_pkt_len_mean`. (Intentional duplicate for CIC compatibility). |
| 41 | `fwd_bytes_bulk_avg` | CIC | $\frac{\text{fbulk\_bytes\_total}}{\text{fbulk\_states}}$ | Ported bulk algorithm (see below). |
| 42 | `fwd_packet_bulk_avg`| CIC | $\frac{\text{fbulk\_pkts\_total}}{\text{fbulk\_states}}$ | Ported bulk algorithm (see below). |
| 43 | `fwd_bulk_rate_avg` | CIC | $\frac{\text{fbulk\_bytes\_total}}{\text{fbulk\_duration}}$ | Ported bulk algorithm (see below). |
| 44 | `bwd_bytes_bulk_avg` | CIC | $\frac{\text{bbulk\_bytes\_total}}{\text{bbulk\_states}}$ | Ported bulk algorithm (see below). |
| 45 | `bwd_packet_bulk_avg`| CIC | $\frac{\text{bbulk\_pkts\_total}}{\text{bbulk\_states}}$ | Ported bulk algorithm (see below). |
| 46 | `bwd_bulk_rate_avg` | CIC | $\frac{\text{bbulk\_bytes\_total}}{\text{bbulk\_duration}}$ | Ported bulk algorithm (see below). |
| 47 | `subflow_fwd_packets`| CIC | $\frac{N_{fwd}}{N_{subflows}}$ | Ported subflow logic (see below). |
| 48 | `subflow_fwd_bytes` | CIC | $\frac{\text{total\_bytes}_{fwd}}{N_{subflows}}$ | Ported subflow logic (see below). |
| 49 | `subflow_bwd_packets`| CIC | $\frac{N_{bwd}}{N_{subflows}}$ | Ported subflow logic (see below). |
| 50 | `subflow_bwd_bytes` | CIC | $\frac{\text{total\_bytes}_{bwd}}{N_{subflows}}$ | Ported subflow logic (see below). |
| 51 | `active_min` | CIC | $\min(\text{ActiveDurations})$ | Ported Active/Idle logic (see below). |
| 52 | `active_mean` | CIC | $\text{mean}(\text{ActiveDurations})$| Ported Active/Idle logic (see below). |
| 53 | `active_max` | CIC | $\max(\text{ActiveDurations})$| Ported Active/Idle logic (see below). |
| 54 | `active_std` | CIC | $\text{std}(\text{ActiveDurations})$ | Sample std (ddof=1). |
| 55 | `idle_min` | CIC | $\min(\text{IdleDurations})$ | Ported Active/Idle logic (see below). |
| 56 | `idle_mean` | CIC | $\text{mean}(\text{IdleDurations})$ | Ported Active/Idle logic (see below). |
| 57 | `idle_max` | CIC | $\max(\text{IdleDurations})$ | Ported Active/Idle logic (see below). |
| 58 | `idle_std` | CIC | $\text{std}(\text{IdleDurations})$ | Sample std (ddof=1). |

---

### Core Algorithms and Approximations (Group A)

#### 0. IAT Calculations
* **Definition:** Inter-arrival time is the elapsed time between consecutive packets.
* **Porting Details:** To handle slightly out-of-order or duplicate timestamp resolution issues in raw trace files, IATs are clamped to a minimum of `0.0` seconds: $\text{IAT}_i = \max(0.0, t_i - t_{i-1})$.

#### 1. Bulk Features (Features 41-46)
* **Definition:** A bulk is a sequence of at least 4 consecutive packets in one direction (forward or backward) where:
  * Each packet has size $> 0$.
  * Inter-packet times are $\le 1.0$ second.
  * No packet in the opposite direction interrupts the sequence.
* **Porting Details:**
  * When a backward packet arrives, it resets the potential forward bulk start helper (`fbulk_start_helper = 0`) if the backward packet timestamp is greater than the forward bulk start helper.
  * If $\ge 4$ packets are matched, they form a valid bulk. The total duration is tracked by summing the intervals.
  * Extracted bulk rates and counts are computed using the formulas in the table.

#### 2. Subflow Features (Features 47-50)
* **Definition:** Subflow boundaries are defined by packet-to-packet idle gaps $> 1.0$ second.
* **Porting Details:**
  * Gaps $> 1.0$s are counted in `sf_count`.
  * If `sf_count == 0` (no gaps $> 1.0$s), the subflow averages return `0.0` as per the Java code `if(sfCount <= 0) return 0;` structure. This ensures compatibility with the original Java implementation.

#### 3. Active/Idle Features (Features 51-58)
* **Definition:** Active periods are periods of active communication. Idle periods are periods with no communication exceeding the activity threshold.
* **Porting Details:**
  * The threshold is set to $5.0$ seconds (porting the Java `activityTimeout` of $5,000,000$ microseconds).
  * If a packet arrives with a gap $> 5.0$s since the last active packet:
    * The active period ends: active duration = `end_active_time - start_active_time` (appended to `active_periods` if $> 0$).
    * An idle period is recorded: idle duration = `current_time - end_active_time` (appended to `idle_periods`).
    * The active state is restarted at the current packet timestamp.
  * At the end of the trace, no final active period is recorded, following the commented out `endActiveIdleTime` method in the Java source.

---

## 2. Group B: 16 WF-specific metadata features

Designed to capture traffic burst shapes specifically for Website Fingerprinting dataset traces.

| # | Feature Name | Group | Formula / Definition | References & Notes |
|---|---|---|---|---|
| 59 | `direction_switch_count` | WF | $\sum_{i=1}^{N-1} [\text{direction}_i \neq \text{direction}_{i-1}]$ | Number of times packet transmission changes direction. |
| 60 | `direction_switch_rate` | WF | $\frac{\text{direction\_switch\_count}}{\max(N_{total} - 1, 1)}$ | Normalized switch count. |
| 61 | `total_burst_count` | WF | count(bursts) | Total number of bursts (maximal runs of same direction). |
| 62 | `fwd_burst_count` | WF | count(bursts with direction = 1) | Outgoing burst count. |
| 63 | `bwd_burst_count` | WF | count(bursts with direction = -1) | Incoming burst count. |
| 64 | `burst_len_mean` | WF | mean(burst packet sizes) | Average number of packets per burst. |
| 65 | `burst_len_std` | WF | std(burst packet sizes) | Sample std (ddof=1) of burst lengths. |
| 66 | `burst_len_max` | WF | max(burst packet sizes) | Maximum burst length. |
| 67 | `fwd_burst_len_mean` | WF | mean(fwd burst packet sizes) | Average packets in outgoing bursts. |
| 68 | `bwd_burst_len_mean` | WF | mean(bwd burst packet sizes) | Average packets in incoming bursts. |
| 69 | `burst_duration_mean` | WF | mean(burst durations) | Burst duration: $t_{burst\_end} - t_{burst\_start}$. |
| 70 | `burst_duration_std` | WF | std(burst durations) | Sample std (ddof=1) of burst durations. |
| 71 | `burst_duration_max` | WF | max(burst durations) | Maximum burst duration. |
| 72 | `inter_burst_gap_mean` | WF | mean(inter-burst gaps) | Gap: $t_{burst_{j+1}\_start} - t_{burst_j\_end}$. |
| 73 | `inter_burst_gap_std` | WF | std(inter-burst gaps) | Sample std (ddof=1) of inter-burst gaps. |
| 74 | `first_burst_direction` | WF | direction of first packet | Returns +1.0 or -1.0. Returns 0 if trace is empty. |
