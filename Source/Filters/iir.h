/*
Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.
*/

#pragma once
#ifndef DSY_IIRFILTER_H
#define DSY_IIRFILTER_H

#include <cstdint>
#include <cstring> // for memset
#include <cassert>
#include <utility>

#ifdef USE_ARM_DSP
#include <arm_math.h> // required for platform-optimized version
#endif

/**   @brief IIR Biquad Filter implementation, generic and ARM CMSIS DSP based
 *    @authors Kien Phan Huy (acousticir.free.fr), Lluc Simó Margalef (acrilique)
 *  Based on fir.h from Alexander Petrov-Savchenko (axp@soft-amp.com)
 *    @date November 2024
 */

namespace daisysp
{

/* use this as a template parameter to indicate user-provided memory storage */
#define IIRFILTER_USER_MEMORY 0, 0

/** Helper class that defines the memory model - internal or user-provided 
 * \param max_size - maximal filter length
 * \param max_block - maximal length of the block processing
 * if both parameters are 0, does NOT allocate any memory and instead
 * requires user-provided memory blocks to be passed as parameters.
 *
 * Not intended to be used directly, so constructor is not exposed
 */
template <size_t max_size, size_t max_block>
struct IIRMemory
{
    /* Public part of the API to be passed through to the FIR users */
  public:
    /* Reset the internal filter state (but not the coefficients) */
    void Reset() { memset(state_, 0, state_size_ * sizeof(state_[0])); }

  protected:
    IIRMemory() : state_{0}, coefs_{0}, size_(0) {}   
    /* Expression for the maximum block size */
    static constexpr size_t MaxBlock() { return max_block; }

    /** Configure the filter coefficients
     * \param coefs - pointer to coefficients (tail-first order)
     * \param size - number of coefficients pointed by coefs (filter length)
     * \param reverse - flag to perform reversing of the filter
     * \return true if all conditions are met and the filter is configured
     */
    bool SetCoefs(const float coefs[], size_t size)
    {
        /* truncate silently */
        size_ = DSY_MIN(size, max_size);
        /* just copy as is */
        memcpy(coefs_, coefs, size_ * sizeof(coefs[0]) * 5); // 5 coefficients per stage
        return true;
    }

    static constexpr size_t state_size_ = max_size * 2;
    float                   state_[state_size_]; /*< Internal state buffer */
    float                   coefs_[max_size * 5];    /*< Filter coefficients - 5 per stage */
    size_t                  size_; /*< Active filter length (<= max_size) */
};

/* Specialization for user-provided memory */
template <>
struct IIRMemory<IIRFILTER_USER_MEMORY>
{
    /* Public part of the API to be passed through to the FIRFilter user */
  public:
    /** Set user-provided state buffer
     * \param state - pointer to the allocated memory block
     * \param length - length of the provided memory block (in elements)
     * The length should be determined as follows 
     * length >= max_filter_size + max_processing_block - 1
     */
    void SetStateBuffer(float state[], size_t length)
    {
        state_ = state;
        state_size_ = length;
    }
    /* Reset the internal filter state (but not the coefficients) */
    void Reset()
    {
        assert(nullptr != state_);
        assert(0 != state_size_);
        if(nullptr != state_)
        {
            memset(state_, 0, state_size_ * sizeof(state_[0]));
        }
    }

  protected:
    IIRMemory() : state_(nullptr), coefs_(nullptr), size_(0), state_size_(0) {}

    /* Expression for the maximum processing block size currently supported */
    size_t MaxBlock() const
    {
        return state_size_ + 1u > size_ ? state_size_ + 1u - size_ : 0;
    }

    /** Configure the filter coefficients
     * \param coefs - pointer to coefficients (tail-first order)
     * \param size - number of filter stages 
     * \param reverse - flag to perform reversing of the filter
     * \return true if all conditions are met and the filter is configured
     */
    bool SetCoefs(const float coefs[], size_t size)
    {
        assert(nullptr != coefs || 0 == size);
        coefs_ = coefs;
        size_ = size;
        return true;
    }

    float*       state_;      /*< Internal state buffer */
    const float* coefs_;      /*< Filter coefficients */
    size_t       size_;       /*< number of filter stages */
    size_t       state_size_; /*< length of the state buffer */
};

/** Generic IIR implementation using Direct Form II Transposed structure
 * \param max_size - maximal number of biquad stages
 * \param max_block - maximal block size for ProcessBlock()
 * if both parameters are 0 (via IIRFILTER_USER_MEMORY macro)
 * Assumes the user will provide own memory buffers
 * Otherwise statically allocates the necessary buffers
 */
template <size_t max_size, size_t max_block>
class IIRFilterImplGeneric : public IIRMemory<max_size, max_block>
{
  private:
    using IIRMem = IIRMemory<max_size, max_block>; // just a shorthand

  public:
    /* Default constructor */
    IIRFilterImplGeneric() {}

    /* Reset filter state (but not the coefficients) */
    using IIRMem::Reset;

    /* IIR Latency is always 0, but API is unified with FFT and fast convolution */
    static constexpr size_t GetLatency() { return 0; }

    /* Process one sample at a time using Direct Form II Transposed structure */
    float Process(float in)
    {
        float out = in;
        
        // Process each biquad stage
        for(size_t stage = 0; stage < size_; stage++)
        {
            const size_t coef_idx = stage * 5;  // 5 coefficients per stage
            const size_t state_idx = stage * 2; // 2 state variables per stage
            
            // Get coefficients for this stage
            const float b0 = coefs_[coef_idx];
            const float b1 = coefs_[coef_idx + 1];
            const float b2 = coefs_[coef_idx + 2];
            const float a1 = coefs_[coef_idx + 3];
            const float a2 = coefs_[coef_idx + 4];
            
            // Direct Form II Transposed implementation
            // Break down computation like ARM implementation for numerical stability
            float p0 = b0 * out;
            float p1 = b1 * out;
            float acc = p0 + state_[state_idx];
            p0 = b2 * out;
            float p3 = a1 * acc;
            float A1 = p1 + p3;
            float p4 = a2 * acc;
            state_[state_idx] = A1 + state_[state_idx + 1];
            state_[state_idx + 1] = p0 + p4;
            
            out = acc;
        }
        
        return out;
    }

    /* Process a block of data */
    void ProcessBlock(const float* pSrc, float* pDst, size_t block)
    {
        assert(block <= IIRMem::MaxBlock());
        assert(size_ > 0u);
        assert(nullptr != pSrc);
        assert(nullptr != pDst);

        // Process each sample in the block
        for(size_t i = 0; i < block; i++)
        {
            pDst[i] = Process(pSrc[i]);
        }
    }

    /** Set filter coefficients
     * Coefficients need to be in the format [b0, b1, b2, a1, a2] for each stage
     */
    bool SetIIR(const float* ir, size_t numStages)
    {
        /* Function order is important */
        const bool result = IIRMem::SetCoefs(ir, numStages);
        Reset();
        return result;
    }

    /* Create an alias to comply with DaisySP API conventions */
    template <typename... Args>
    inline auto Init(Args&&... args)
        -> decltype(SetIIR(std::forward<Args>(args)...))
    {
        return SetIIR(std::forward<Args>(args)...);
    }

  protected:
    using IIRMem::coefs_;      /*< IIR coefficients buffer or pointer */
    using IIRMem::size_;       /*< IIR length*/
    using IIRMem::state_;      /*< IIR state buffer or pointer */
};

#if(defined(USE_ARM_DSP) && defined(__arm__))
/** ARM-specific IIR implementation, expose only on __arm__ platforms
 * \param max_size - maximal filter length
 * \param max_block - maximal block size for ProcessBlock()
 * if both parameters are 0 (via IIRFILTER_USER_MEMORY macro)
 * Assumes the user will provide own memory buffers
 * Otherwise statically allocates the necessary buffers
 */
template <size_t max_size, size_t max_block>
class IIRFilterImplARM : public IIRMemory<max_size, max_block>
{
  private:
    using IIRMem = IIRMemory<max_size, max_block>; // just a shorthand

  public:
    /* Default constructor */
    IIRFilterImplARM() : iir_{0} {}

    /* FIR Latency is always 0, but API is unified with FFT and FastConv */
    static constexpr size_t GetLatency() { return 0; }

    /* Process one sample at a time */
    float Process(float in)
    {
        float out;
        arm_biquad_cascade_df2T_f32(&iir_, &in, &out, 1);
        return out;
    }

    /* Process a block of data */
    void ProcessBlock(float* pSrc, float* pDst, size_t block)
    {
        arm_biquad_cascade_df2T_f32(&iir_, pSrc, pDst, block);
    }

    /** Set filter coefficients
     * Coefficients need to be in the format [b0, b1, b2, a1, a2] for each stage
     */
    bool SetIIR(const float* ir, size_t numStages)
    {
        /* Function order is important */
        const bool result = IIRMem::SetCoefs(ir, numStages);
        arm_biquad_cascade_df2T_init_f32(&iir_, numStages, (float*)coefs_, state_);
        return result;
    }

    /* Create an alias to comply with DaisySP API conventions */
    template <typename... Args>
    inline auto Init(Args&&... args)
        -> decltype(SetIIR(std::forward<Args>(args)...))
    {
        return SetIIR(std::forward<Args>(args)...);
    }

  protected:
    arm_biquad_cascade_df2T_instance_f32 iir_; /*< ARM CMSIS DSP library IIR filter instance */ 
    using IIRMem::coefs_;      /*< IIR coefficients buffer or pointer */
    using IIRMem::size_;       /*< IIR length*/
    using IIRMem::state_;      /*< IIR state buffer or pointer */
};

/* default to ARM implementation */
template <size_t max_size, size_t max_block>
using IIR = IIRFilterImplARM<max_size, max_block>;

#else // USE_ARM_DSP

/* default to generic implementation */
template <size_t max_size, size_t max_block>
using IIR = IIRFilterImplGeneric<max_size, max_block>;

#endif // USE_ARM_DSP 

} // namespace daisysp

#endif // DSY_IIRFILTER_H
