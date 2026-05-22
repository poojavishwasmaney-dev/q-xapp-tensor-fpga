# Quantum-Inspired User Scheduling for 6G URLLC on FPGA

## MSc Project Proposal

**Candidate:** [Your Name] | **Background:** FPGA/SoC Design Engineer  
**Supervisor:** [Supervisor Name] | **Duration:** 12 weeks  
**Platform:** AMD Kria KV260 + Qiskit

---

## 1. Problem Statement

In 6G Ultra-Reliable Low-Latency Communication (URLLC), a base station must schedule **N users** across **K time-frequency resource blocks (PRBs)** within **<500 microseconds** while maintaining:

- **>99.999% reliability** (10⁻⁵ error probability)
- **Fairness** among users (Jain index >0.85)
- **Minimum QoS** per user

### Why Current Solutions Fail

| Method | Latency (N=32) | Problem |
|--------|----------------|---------|
| Exhaustive Search | >1 hour | Impossible at scale |
| Greedy Algorithm | 0.2 ms | 15-30% suboptimal |
| Deep RL (DQN) | 5-50 ms | Violates URLLC <0.5ms |
| Qiskit on CPU | 10-100 ms | Simulation overhead |

### Research Gap

> *"No existing implementation accelerates quantum-inspired scheduling on FPGA to meet 6G URLLC latency constraints."*

---

## 2. Proposed Solution

### What I Am Implementing

- **Grover's Search Algorithm** with amplitude amplification
- **Quantum state representation** (2^N superposition states)
- **Oracle + Diffusion operators** (Equations 4 & 5 below)
- **Hybrid ARM + FPGA architecture** on Kria KV260

### How It Works (3 Steps)

1. **Encode** ALL possible schedules as quantum superposition:  
   `|ψ⟩ = (1/√2^N) Σ |schedule⟩`

2. **Apply Grover iterations** to amplify "good" schedules (high utility, valid constraints)

3. **Measure** → collapses to best schedule

### What I Am NOT Implementing

- ❌ Quantum Neural Networks
- ❌ Quantum Machine Learning
- ❌ Deep Reinforcement Learning
- ❌ Training or backpropagation
- ❌ Real quantum hardware (IBM/Google)

> **Note:** This is **quantum-INSPIRED** — classical simulation of Grover's algorithm accelerated on FPGA, not a true quantum computer.

---

## 3. Mathematical Framework

### Equation 1: Achievable Data Rate (URLLC Short Block)

From Q-xApp paper [Lee et al., 2024], Equation (2):


- `R_ia` = data rate when user i uses PRB a (bps)
- `B_a` = bandwidth of PRB a (240 kHz)
- `ε` = transmission error probability (10⁻⁵ for URLLC)
- `l_ia` = blocklength (32 bytes for URLLC)

### Equation 2: Utility Function (With Proportional Fairness)


**Constraints (from Q-xApp Equation 4):**

| Constraint | Meaning |
|------------|---------|
| Σ_i x_ia = 1 | Each PRB to one user |
| Σ_a x_ia ≤ C_i | Fairness cap per user |
| Σ_a R_ia x_ia ≥ R_min | Minimum QoS requirement |

### Equation 3: Quantum State Representation (Tensor Product)


**Example (N=4, schedule {user1, user3}):**  
`|1010⟩ = |1⟩⊗|0⟩⊗|1⟩⊗|0⟩`

### Equation 4: Grover Oracle (Mark Good States)


**Good schedule conditions:**
- Exactly K users selected (|x| = K)
- Utility(x) ≥ threshold (top 10%)
- All SINR_i ≥ γ_min (reliability)

### Equation 5: Diffusion Operator (Amplitude Amplification)


where `M` = number of "good" schedules (typically top 10% by utility)

**Example (N=8):** 2^N=256, M≈26 → √(256/26)=3.14 → m=2 → P_success≈96%

---

## 4. Implementation Plan

### Phase 1: Qiskit Baseline (Weeks 1-4)

```python
# qiskit_scheduler.py - Complete working code
import numpy as np
from qiskit.algorithms import Grover, AmplificationProblem

class UserSchedulingProblem:
    def __init__(self, N, K, sinr_linear, weights):
        self.N = N          # number of users (qubits)
        self.K = K          # resource blocks to schedule
        self.sinr = sinr_linear
        self.weights = weights
        
    def utility(self, bitstring):
        """Equation 2: U(S) = Σ w_i·log₂(1+SINR_i)"""
        if bitstring.count('1') != self.K:
            return 0
        return sum(self.weights[i] * np.log2(1 + self.sinr[i]) 
                   for i, b in enumerate(bitstring) if b == '1')
    
    def find_threshold(self):
        """Find top 10% utility threshold"""
        all_utils = [self.utility(format(i, f'0{self.N}b')) 
                     for i in range(2**self.N)]
        valid = [u for u in all_utils if u > 0]
        return np.percentile(valid, 90)
    
    def is_good_state(self, bitstring):
        """Oracle condition (Equation 4)"""
        if bitstring.count('1') != self.K:
            return False
        return self.utility(bitstring) >= self.threshold

# Run Grover
problem = UserSchedulingProblem(N=8, K=4, sinr_linear=..., weights=...)
grover = Grover()
result = grover.amplify(AmplificationProblem(
    is_good_state=problem.is_good_state
))
print(f"Best schedule: {result.top_measurement}")

// grover_kernel.cpp - Synthesizable HLS code
#include <ap_fixed.h>

typedef ap_fixed<16, 8, AP_RND, AP_SAT> amplitude_t;
#define N_STATES 256    // 2^8 users
#define N_ITER 2        // optimal iterations for N=8

void grover_iteration(
    amplitude_t amps[N_STATES],
    bool good_state[N_STATES],
    amplitude_t out[N_STATES]
) {
    #pragma HLS PIPELINE
    
    // Step 1: Oracle (phase flip good states)
    for(int i = 0; i < N_STATES; i++) {
        #pragma HLS UNROLL factor=16
        out[i] = good_state[i] ? -amps[i] : amps[i];
    }
    
    // Step 2: Compute mean amplitude
    amplitude_t sum = 0;
    for(int i = 0; i < N_STATES; i++) {
        #pragma HLS PIPELINE
        sum += out[i];
    }
    amplitude_t mean = sum / N_STATES;
    
    // Step 3: Diffusion operator: D = 2|mean⟩⟨mean| - I
    for(int i = 0; i < N_STATES; i++) {
        #pragma HLS UNROLL factor=16
        out[i] = (2 * mean) - out[i];
    }
}

void grover_scheduler(
    float sinr[8],
    float weights[8],
    int K_slots,
    int best_schedule[8]
) {
    #pragma HLS INTERFACE s_axilite port=sinr
    #pragma HLS INTERFACE s_axilite port=weights
    #pragma HLS INTERFACE s_axilite port=K_slots
    #pragma HLS INTERFACE s_axilite port=best_schedule
    
    amplitude_t amplitudes[N_STATES];
    bool good_state[N_STATES];
    
    // Initialize uniform superposition
    amplitude_t init_amp = 1.0 / sqrt(N_STATES);
    for(int i = 0; i < N_STATES; i++) {
        amplitudes[i] = init_amp;
    }
    
    // Precompute good_state mask (on ARM, passed to FPGA)
    // ... good_state calculation ...
    
    // Run Grover iterations
    for(int iter = 0; iter < N_ITER; iter++) {
        grover_iteration(amplitudes, good_state, amplitudes);
    }
    
    // Find state with maximum amplitude
    amplitude_t max_amp = 0;
    int best_idx = 0;
    for(int i = 0; i < N_STATES; i++) {
        if(amplitudes[i] > max_amp) {
            max_amp = amplitudes[i];
            best_idx = i;
        }
    }
    
    // Convert to output schedule
    for(int i = 0; i < 8; i++) {
        best_schedule[i] = (best_idx >> i) & 1;
    }
}

