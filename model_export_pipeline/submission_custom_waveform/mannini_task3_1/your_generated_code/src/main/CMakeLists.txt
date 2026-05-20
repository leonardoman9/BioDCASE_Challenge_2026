idf_component_register(
    SRCS main.cpp metrics.cpp model.cpp esp_micro_profiler.cpp frontend_frozen.cpp frontend_frozen_data.cpp
    PRIV_REQUIRES spi_flash driver esp_timer
    INCLUDE_DIRS "")

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-maybe-uninitialized
    -Wno-missing-field-initializers
    -Wno-error=sign-compare
    -Wno-error=double-promotion
    -Wno-type-limits)
