#pragma once
/**
 * map_runner.hpp — MapTR 推理 C++ 包装器
 *
 * 通过 subprocess 调用 Python maptr_model_infer 执行 MapTR 推理，
 * 并将结果解析为 MapResult 结构体，供联合感知主程序使用。
 *
 * 支持三种模式（与 Python 端 MapInference 对应）：
 *   "model"      — MapTRv2 神经网络（需 maptr conda 环境 + checkpoint）
 *   "gt"         — NuScenes GT 矢量地图
 *   "trajectory" — 自车轨迹降级
 *   "auto"       — 按 model > gt > trajectory 自动选择
 *
 * 使用示例：
 * @code
 *   fastbev::perception::MapRunnerConfig cfg;
 *   cfg.mode         = "gt";
 *   cfg.nuscenes_dir = "data/nuscenes";
 *   cfg.frames_dir   = "outputs/frames";
 *
 *   fastbev::perception::MapRunner runner(cfg);
 *
 *   // 批量推理（将 map_result.json 写入每帧目录）
 *   int n = runner.run_batch(true);   // overwrite=true
 *
 *   // 读取单帧结果（已有 map_result.json）
 *   fastbev::perception::MapResult result;
 *   runner.read_result("outputs/frames/frame_00000", result);
 * @endcode
 */

#include "perception_types.hpp"
#include <string>
#include <functional>

namespace fastbev {
namespace perception {

// ─── MapRunner 配置 ────────────────────────────────────────────────────────

struct MapRunnerConfig {
    // 推理模式："model" | "gt" | "trajectory" | "auto"
    std::string mode          = "auto";

    // 帧目录（run_batch 时遍历此目录下的 frame_* 子目录）
    std::string frames_dir    = "outputs/frames";

    // NuScenes 数据根目录（gt 模式需要）
    std::string nuscenes_dir  = "data/nuscenes";
    std::string nuscenes_ver  = "v1.0-mini";

    // MapTRv2 权重（model 模式需要）
    std::string ckpt_path     = "model/maptr/maptr_nano_r18_110e.pth";
    std::string cfg_path      = "model/maptr/config/maptr_nano_r18_110e.py";

    // 置信度阈值
    float  score_thr          = 0.3f;

    // conda 环境名（Python worker 运行于此环境）
    std::string bev_env       = "bev";

    // 是否在每次 run_batch 前检查 Python 环境
    bool   check_env          = true;

    // verbose
    bool   verbose            = true;
};

// ─── MapRunner ─────────────────────────────────────────────────────────────

class MapRunner {
public:
    explicit MapRunner(const MapRunnerConfig& cfg = MapRunnerConfig{});
    ~MapRunner() = default;

    /**
     * 批量推理：对 frames_dir 下所有 frame_* 目录执行地图推理。
     * 每帧结果写入 <frame_dir>/map_result.json。
     *
     * @param overwrite  若 map_result.json 已存在，是否覆盖（false = 跳过）
     * @return 成功处理的帧数；负值表示严重错误
     */
    int run_batch(bool overwrite = false);

    /**
     * 单帧推理：对指定帧目录执行地图推理。
     * （通过 Python 批量模式仅处理该目录，效率低于直接调用 Python；
     *  若已有 map_result.json 则直接读取。）
     *
     * @param frame_dir  帧目录路径
     * @param result     [out] 解析后的地图结果
     * @param overwrite  是否覆盖已有结果
     * @return true 表示成功
     */
    bool run_single(const std::string& frame_dir,
                    MapResult&         result,
                    bool               overwrite = false);

    /**
     * 读取已有的 map_result.json（不重新推理）。
     *
     * @param frame_dir  帧目录路径
     * @param result     [out] 解析后的地图结果
     * @return true 表示成功
     */
    bool read_result(const std::string& frame_dir, MapResult& result) const;

    /**
     * 检查 Python 环境是否满足要求（conda env + checkpoint + compiled ops）。
     * @return true 表示可以运行
     */
    bool check_available() const;

    const MapRunnerConfig& config() const { return cfg_; }

private:
    MapRunnerConfig cfg_;

    // 调用 Python 脚本（返回 exit code）
    int _run_python(const std::string& extra_args) const;

    // 从 JSON 字符串解析 MapResult
    static bool _parse_json(const std::string& json_str,
                             MapResult&         result);

    // 提取 JSON 中的数组元素（极简解析，避免引入第三方库）
    static bool _read_json_file(const std::string& path, std::string& out);
};

}  // namespace perception
}  // namespace fastbev
