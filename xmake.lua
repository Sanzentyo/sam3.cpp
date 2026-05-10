set_project("sam3.cpp")
set_version("1.0.0")
set_xmakever("2.9.0")
set_languages("c++23")
set_warnings("all", "extra")
set_policy("build.warning", true)
set_policy("build.ccache", true)
set_policy("build.optimization.lto", false)

add_rules("mode.debug", "mode.release", "mode.releasedbg", "mode.profile")

if is_plat("macosx", "linux") then
    add_cxflags("-fno-omit-frame-pointer", "-fdiagnostics-color=always")
end
if is_plat("windows") then
    add_defines("NOMINMAX", "WIN32_LEAN_AND_MEAN", "_CRT_SECURE_NO_WARNINGS")
    set_runtimes("MD")
end

option("cuda")
    set_default(true)
    set_showmenu(true)
    set_description("Enable the ggml CUDA backend")
option_end()

option("cuda_arch")
    set_default("120")
    set_showmenu(true)
    set_description("CMAKE_CUDA_ARCHITECTURES value used when CUDA is enabled")
option_end()

option("cuda_host_compiler")
    set_default("/usr/bin/g++-14")
    set_showmenu(true)
    set_description("CMAKE_CUDA_HOST_COMPILER value used when CUDA is enabled")
option_end()

option("cxx_compiler")
    set_default("/usr/bin/g++-14")
    set_showmenu(true)
    set_description("CMAKE_CXX_COMPILER value used for sam3.cpp and examples")
option_end()

option("build_examples")
    set_default(true)
    set_showmenu(true)
    set_description("Build sam3.cpp examples")
option_end()

local function cmake_builddir()
    local mode = get_config("mode") or "release"
    local cuda_suffix = has_config("cuda") and "cuda" or "cpu"
    return path.join(os.projectdir(), "build", "xmake-" .. mode .. "-" .. cuda_suffix)
end

local function cmake_build_type()
    local mode = get_config("mode") or "release"
    if mode == "debug" then
        return "Debug"
    end
    return "Release"
end

local function cmake_configure_args(builddir)
    local args = {
        "-S", os.projectdir(),
        "-B", builddir,
        "-DCMAKE_BUILD_TYPE=" .. cmake_build_type(),
        "-DCMAKE_CXX_STANDARD=23",
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
        "-DCMAKE_CXX_EXTENSIONS=OFF",
        "-DSAM3_BUILD_EXAMPLES=" .. (has_config("build_examples") and "ON" or "OFF"),
        "-DSAM3_BUILD_TESTS=OFF"
    }

    local cxx = get_config("cxx_compiler")
    if cxx and cxx ~= "" and os.isfile(cxx) then
        table.insert(args, "-DCMAKE_CXX_COMPILER=" .. cxx)
    end
    if has_config("cuda") then
        table.insert(args, "-DGGML_CUDA=ON")
        table.insert(args, "-DGGML_CUDA_GRAPHS=ON")
        table.insert(args, "-DGGML_CUDA_FA=ON")
        table.insert(args, "-DCMAKE_CUDA_ARCHITECTURES=" .. get_config("cuda_arch"))
        local host = get_config("cuda_host_compiler")
        if host and host ~= "" and os.isfile(host) then
            table.insert(args, "-DCMAKE_CUDA_HOST_COMPILER=" .. host)
        end
    else
        table.insert(args, "-DGGML_CUDA=OFF")
    end

    return args
end

target("sam3_benchmark")
    set_kind("phony")
    on_build(function (target)
        local builddir = cmake_builddir()
        os.mkdir(builddir)
        os.execv("cmake", cmake_configure_args(builddir))
        os.execv("cmake", {"--build", builddir, "--target", "sam3_benchmark", "-j", tostring(os.default_njob())})
    end)
    on_run(function (target)
        local exe = path.join(cmake_builddir(), "examples", "sam3_benchmark")
        os.execv(exe, get_config("arguments") or {})
    end)

target("sam3_smoke")
    set_kind("phony")
    on_build(function (target)
        local builddir = cmake_builddir()
        os.mkdir(builddir)
        os.execv("cmake", cmake_configure_args(builddir))
        os.execv("cmake", {"--build", builddir, "--target", "sam3_smoke", "-j", tostring(os.default_njob())})
    end)
    on_run(function (target)
        local exe = path.join(cmake_builddir(), "examples", "sam3_smoke")
        os.execv(exe, get_config("arguments") or {})
    end)

target("sam3_profile_edgetam")
    set_kind("phony")
    on_build(function (target)
        local builddir = cmake_builddir()
        os.mkdir(builddir)
        os.execv("cmake", cmake_configure_args(builddir))
        os.execv("cmake", {"--build", builddir, "--target", "sam3_profile_edgetam", "-j", tostring(os.default_njob())})
    end)

target("sam3")
    set_kind("phony")
    on_build(function (target)
        local builddir = cmake_builddir()
        os.mkdir(builddir)
        os.execv("cmake", cmake_configure_args(builddir))
        os.execv("cmake", {"--build", builddir, "--target", "sam3", "-j", tostring(os.default_njob())})
    end)
