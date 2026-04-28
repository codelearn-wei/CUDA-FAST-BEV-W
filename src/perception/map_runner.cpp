/**
 * map_runner.cpp — MapTR 推理 C++ 包装器实现
 */

#include "map_runner.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace fastbev {
namespace perception {

// ─── 内部辅助 ─────────────────────────────────────────────────────────────

static bool _exists(const std::string& p)
{
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

// ─── 构造 ─────────────────────────────────────────────────────────────────

MapRunner::MapRunner(const MapRunnerConfig& cfg)
    : cfg_(cfg)
{}

// ─── check_available ──────────────────────────────────────────────────────

bool MapRunner::check_available() const
{
    // 通过 Python 脚本本身检查（run_maptr.py --check-model 会打印状态并退出 0/1）
    std::string cmd =
        "conda run -n " + cfg_.bev_env +
        " python tools/maptr/run_maptr.py --check-model 2>&1";
    int ret = std::system(cmd.c_str());
    return ret == 0;
}

// ─── _run_python ──────────────────────────────────────────────────────────

int MapRunner::_run_python(const std::string& extra_args) const
{
    // 调用 tools/maptr/run_maptr.py（在 bev 环境中运行）
    std::string cmd =
        "conda run -n " + cfg_.bev_env +
        " python tools/maptr/run_maptr.py"
        " --frames-dir "   + cfg_.frames_dir  +
        " --nuscenes-dir " + cfg_.nuscenes_dir +
        " --mode "         + cfg_.mode         +
        " --score-thr "    + std::to_string(cfg_.score_thr);

    // model 模式需传入 checkpoint 路径（config 由 Python 端默认推断）
    if (cfg_.mode == "model" || cfg_.mode == "auto") {
        if (!cfg_.ckpt_path.empty())
            cmd += " --ckpt " + cfg_.ckpt_path;
    }

    if (!extra_args.empty())
        cmd += " " + extra_args;

    if (cfg_.verbose)
        printf("[MapRunner] 执行: %s\n", cmd.c_str());

    return std::system(cmd.c_str());
}

// ─── run_batch ────────────────────────────────────────────────────────────

int MapRunner::run_batch(bool overwrite)
{
    if (!_exists(cfg_.frames_dir)) {
        fprintf(stderr, "[MapRunner] frames_dir 不存在: %s\n",
                cfg_.frames_dir.c_str());
        return -1;
    }

    std::string extra = overwrite ? "--overwrite" : "";
    int ret = _run_python(extra);
    if (ret != 0) {
        fprintf(stderr, "[MapRunner] Python 推理失败（exit=%d）\n", ret);
        return -1;
    }

    // 统计成功帧数
    int count = 0;
    // 遍历 frames_dir/frame_XXXXX 子目录
    for (int i = 0; ; ++i) {
        char sub[256];
        snprintf(sub, sizeof(sub), "%s/frame_%05d", cfg_.frames_dir.c_str(), i);
        if (!_exists(sub)) break;
        std::string res = std::string(sub) + "/map_result.json";
        if (_exists(res)) ++count;
    }
    if (cfg_.verbose)
        printf("[MapRunner] 批量推理完成，成功帧数: %d\n", count);
    return count;
}

// ─── run_single ───────────────────────────────────────────────────────────

bool MapRunner::run_single(const std::string& frame_dir,
                            MapResult&         result,
                            bool               overwrite)
{
    std::string res_path = frame_dir + "/map_result.json";

    // 如果结果已存在且不需要覆盖，直接读取
    if (!overwrite && _exists(res_path))
        return read_result(frame_dir, result);

    // 临时设置 frames_dir 为 frame_dir 的父目录
    // 并通过 --overwrite 只处理该帧（run_maptr.py 支持按 frame_dir 过滤）
    // 此处简化：调用 Python 对整个 frames_dir 推理，然后读取该帧结果
    // 适合离线验证；在线场景建议直接调用 Python worker 接口
    std::string extra = overwrite ? "--overwrite" : "";
    int ret = _run_python(extra);
    if (ret != 0) {
        fprintf(stderr, "[MapRunner] run_single: Python 推理失败\n");
        return false;
    }
    return read_result(frame_dir, result);
}

// ─── read_result ──────────────────────────────────────────────────────────

bool MapRunner::read_result(const std::string& frame_dir,
                             MapResult&         result) const
{
    std::string path = frame_dir + "/map_result.json";
    std::string json_str;
    if (!_read_json_file(path, json_str)) {
        if (cfg_.verbose)
            fprintf(stderr, "[MapRunner] 无法读取: %s\n", path.c_str());
        return false;
    }
    result.raw_json = json_str;
    return _parse_json(json_str, result);
}

// ─── _read_json_file ──────────────────────────────────────────────────────

bool MapRunner::_read_json_file(const std::string& path, std::string& out)
{
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return !out.empty();
}

// ─── _parse_json ──────────────────────────────────────────────────────────
//
// 极简 JSON 解析：仅提取 MapResult 所需字段，不依赖第三方库。
// map_result.json 格式（来自 Python MapInference）：
// {
//   "source": "gt",
//   "timestamp": 1.23e9,
//   "elements": [
//     { "type": 0, "score": 0.9, "pts": [[x,y], ...] },
//     ...
//   ]
// }

bool MapRunner::_parse_json(const std::string& s, MapResult& result)
{
    // 提取 source
    {
        size_t kp = s.find("\"source\"");
        if (kp != std::string::npos) {
            size_t q1 = s.find('"', kp + 8);
            if (q1 != std::string::npos) {
                size_t q2 = s.find('"', q1 + 1);
                if (q2 != std::string::npos)
                    result.source = s.substr(q1 + 1, q2 - q1 - 1);
            }
        }
    }

    // 提取 timestamp
    {
        size_t kp = s.find("\"timestamp\"");
        if (kp != std::string::npos) {
            size_t vp = s.find_first_of("-0123456789.", kp + 11);
            if (vp != std::string::npos) {
                try { result.timestamp = std::stod(s.substr(vp)); }
                catch (...) {}
            }
        }
    }

    // 提取 elements 数组
    // 每个 element 形如：{ "type": N, "score": F, "pts": [[x,y], ...] }
    result.elements.clear();
    size_t arr_start = s.find("\"elements\"");
    if (arr_start == std::string::npos) return true; // 空结果（有效）

    arr_start = s.find('[', arr_start);
    if (arr_start == std::string::npos) return true;

    // 逐个 { } 块解析
    size_t pos = arr_start + 1;
    while (true) {
        size_t blk_s = s.find('{', pos);
        if (blk_s == std::string::npos) break;

        // 找到匹配的 }
        int depth = 1;
        size_t blk_e = blk_s + 1;
        while (blk_e < s.size() && depth > 0) {
            if      (s[blk_e] == '{') ++depth;
            else if (s[blk_e] == '}') --depth;
            ++blk_e;
        }
        if (depth != 0) break;

        std::string blk = s.substr(blk_s, blk_e - blk_s);

        MapElement elem;

        // type
        {
            size_t tp = blk.find("\"type\"");
            if (tp != std::string::npos) {
                size_t vp = blk.find_first_of("0123456789", tp + 6);
                if (vp != std::string::npos) {
                    try {
                        elem.type = static_cast<MapElementType>(std::stoi(blk.substr(vp)));
                    } catch (...) {}
                }
            }
        }

        // score
        {
            size_t tp = blk.find("\"score\"");
            if (tp != std::string::npos) {
                size_t vp = blk.find_first_of("-0123456789.", tp + 7);
                if (vp != std::string::npos) {
                    try { elem.score = std::stof(blk.substr(vp)); }
                    catch (...) {}
                }
            }
        }

        // pts: [[x0,y0],[x1,y1],...]
        {
            size_t pp = blk.find("\"pts\"");
            if (pp != std::string::npos) {
                size_t arr_p = blk.find('[', pp + 5);
                if (arr_p != std::string::npos) {
                    // 解析内层 [x,y] 对
                    size_t p2 = arr_p + 1;
                    while (true) {
                        size_t inner_s = blk.find('[', p2);
                        if (inner_s == std::string::npos) break;
                        size_t inner_e = blk.find(']', inner_s);
                        if (inner_e == std::string::npos) break;
                        std::string pair_str = blk.substr(inner_s + 1,
                                                           inner_e - inner_s - 1);
                        // 解析 x
                        size_t vp = pair_str.find_first_of("-0123456789.");
                        if (vp != std::string::npos) {
                            try {
                                float x = std::stof(pair_str.substr(vp));
                                size_t cp = pair_str.find(',', vp);
                                float y = 0.f;
                                if (cp != std::string::npos) {
                                    size_t vp2 = pair_str.find_first_of("-0123456789.", cp);
                                    if (vp2 != std::string::npos)
                                        y = std::stof(pair_str.substr(vp2));
                                }
                                elem.points.push_back({x, y});
                            } catch (...) {}
                        }
                        p2 = inner_e + 1;
                    }
                }
            }
        }

        result.elements.push_back(std::move(elem));
        pos = blk_e;
    }

    return true;
}

}  // namespace perception
}  // namespace fastbev
