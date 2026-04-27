/**
 * stb_image_impl.cpp — 为整个项目提供 stb_image 函数实现
 *
 * 此文件仅包含一次，生成供 libfastbev_core.so 导出的 stbi_load 等符号。
 * 其他编译单元直接 #include <stb_image.h>（不带 IMPL 宏）即可使用这些函数。
 */

// stb_image_write 实现保留在 main.cpp 中（仅在可执行文件中需要）
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
