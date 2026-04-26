#pragma once
/**
 * multi_camera_buffer.hpp — 多相机帧缓冲与时间同步
 *
 * 提供一个简单的环形缓冲，用于：
 *   1. 存储最近 N 帧的多相机数据
 *   2. 支持按帧 ID 或时间戳检索
 *   3. 在在线场景中缓存等待所有相机就绪的帧（近似时间同步）
 */

#include "camera_frame.hpp"
#include <deque>
#include <mutex>
#include <functional>

namespace fastbev {
namespace camera {

// ─── 缓冲配置 ──────────────────────────────────────────────────────────────

struct BufferConfig {
    int    max_frames    = 10;     // 最多缓存的帧数
    double sync_slop_s   = 0.05;  // 时间同步容差（秒）
    bool   drop_oldest   = true;  // 满时丢弃最旧帧
};

// ─── 多相机帧缓冲 ──────────────────────────────────────────────────────────

class MultiCameraBuffer {
public:
    using FramePtr = std::unique_ptr<CameraFrame>;
    using FrameCallback = std::function<void(const CameraFrame&)>;

    explicit MultiCameraBuffer(const BufferConfig& cfg = BufferConfig{})
        : cfg_(cfg) {}

    /**
     * 推入一帧（离线模式：完整帧直接入队）
     */
    void push(FramePtr frame) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (static_cast<int>(buf_.size()) >= cfg_.max_frames && cfg_.drop_oldest)
            buf_.pop_front();
        buf_.push_back(std::move(frame));
    }

    /**
     * 弹出最旧的帧（FIFO）
     */
    FramePtr pop() {
        std::lock_guard<std::mutex> lk(mtx_);
        if (buf_.empty()) return nullptr;
        auto f = std::move(buf_.front());
        buf_.pop_front();
        return f;
    }

    /**
     * 查看最旧帧（不出队）
     */
    const CameraFrame* peek() const {
        std::lock_guard<std::mutex> lk(mtx_);
        if (buf_.empty()) return nullptr;
        return buf_.front().get();
    }

    int  size()  const { std::lock_guard<std::mutex> lk(mtx_); return static_cast<int>(buf_.size()); }
    bool empty() const { return size() == 0; }
    void clear() { std::lock_guard<std::mutex> lk(mtx_); buf_.clear(); }

private:
    BufferConfig cfg_;
    mutable std::mutex mtx_;
    std::deque<FramePtr> buf_;
};

}  // namespace camera
}  // namespace fastbev
