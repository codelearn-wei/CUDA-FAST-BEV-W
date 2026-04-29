#pragma once
/**
 * prefetch_loader.hpp — 帧预取加载器（双缓冲）
 *
 * 在 BEV Worker 处理第 N 帧的同时，后台线程异步预加载第 N+1 帧，
 * 将磁盘 I/O（6 路 JPEG + 3 个几何张量，共 ~25MB）从 BEV 关键路径中隐藏。
 *
 * 使用方式（run_pipeline.cpp BEV 主循环）：
 * @code
 *   PrefetchLoader prefetch;
 *   // 在处理帧 N 时，提前启动帧 N+1 的后台加载
 *   prefetch.submit(frame_dirs[i+1], i+1);   // 非阻塞
 *   // 当需要帧 N+1 时
 *   camera::CameraFrame frame;
 *   RawImageInput map_input;
 *   prefetch.consume(frame, map_input);       // 若已加载则立即返回，否则等待
 * @endcode
 *
 * 线程安全：submit() 和 consume() 不能同时调用（由 run_pipeline 单线程调用）。
 */

#include "image_preprocessor.hpp"
#include "../camera/camera_frame.hpp"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <string>
#include <memory>
#include <cstdio>

namespace fastbev {
namespace perception {

class PrefetchLoader {
public:
    PrefetchLoader() = default;

    ~PrefetchLoader() {
        // 等待后台任务完成，释放资源
        _wait_done();
    }

    /**
     * 提交后台预加载任务（非阻塞）。
     * 若上一帧预加载尚未消费，等待其完成后再启动新任务（不丢帧）。
     *
     * @param frame_dir  帧目录路径
     * @param frame_id   帧 ID（用于填充 CameraFrame::frame_id）
     */
    void submit(const std::string& frame_dir, uint64_t frame_id) {
        _wait_done();  // 确保没有未消费的旧结果

        state_    = State::LOADING;
        ok_       = false;
        // 在后台线程中执行 ImagePreprocessor::from_frame_dir
        bg_thread_ = std::thread([this, frame_dir, frame_id]() {
            RawImageInput  map_inp;
            camera::CameraFrame frm;
            bool result = ImagePreprocessor::from_frame_dir(
                frame_dir, frame_id, frm, map_inp);
            {
                std::unique_lock<std::mutex> lk(mtx_);
                if (result) {
                    ready_frame_ = std::move(frm);
                    ready_map_   = std::move(map_inp);
                    ok_          = true;
                }
                state_ = State::READY;
            }
            cv_.notify_one();
        });
    }

    /**
     * 消费已预加载的帧（阻塞直到帧就绪）。
     * 消费后状态重置为 IDLE，可以再次 submit。
     *
     * @return true  = 帧加载成功
     *         false = 帧加载失败（from_frame_dir 返回 false）
     */
    bool consume(camera::CameraFrame& out_frame,
                 RawImageInput&       out_map)
    {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this]{ return state_ == State::READY; });
        }
        if (bg_thread_.joinable())
            bg_thread_.join();

        if (!ok_) {
            state_ = State::IDLE;
            return false;
        }

        out_frame = std::move(ready_frame_);
        out_map   = std::move(ready_map_);
        state_    = State::IDLE;
        return true;
    }

    /**
     * 是否有预加载任务正在运行或已就绪（即不处于 IDLE 状态）。
     */
    bool is_busy() const { return state_ != State::IDLE; }

private:
    enum class State { IDLE, LOADING, READY };

    void _wait_done() {
        if (state_ == State::IDLE) return;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this]{ return state_ != State::LOADING; });
        }
        if (bg_thread_.joinable())
            bg_thread_.join();
        state_ = State::IDLE;
    }

    State                state_ = State::IDLE;
    bool                 ok_    = false;
    std::thread          bg_thread_;
    std::mutex           mtx_;
    std::condition_variable cv_;
    camera::CameraFrame  ready_frame_;
    RawImageInput        ready_map_;
};

}  // namespace perception
}  // namespace fastbev
