add_library(culib STATIC
    utils.cu
)

target_compile_options(culib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --use_fast_math
                       --generate-line-info
                       >)