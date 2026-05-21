#pragma once

extern "C" {
#include "numtoy.h"
}
#include <string>
#include <vector>
#include <stdexcept>
#include <utility>

namespace numtoy {

enum class DataType : uint32_t {
    Float = 0,
    Int = 1,
    DynamicFloat = 2,
    FloatingInt = 3
};

class Engine {
public:
    Engine() {
        ptr_ = nt_engine_new();
        if (!ptr_) {
            throw std::runtime_error("Failed to initialize NumToy hardware engine");
        }
    }

    ~Engine() {
        if (ptr_) {
            nt_engine_free(ptr_);
        }
    }

    // Disable copying
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Allow moving
    Engine(Engine&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    Engine& operator=(Engine&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                nt_engine_free(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    HardwareEngine* get() const { return ptr_; }

private:
    HardwareEngine* ptr_ = nullptr;
};

class Expr {
public:
    explicit Expr(::Expr* ptr) : ptr_(ptr) {}

    ~Expr() {
        if (ptr_) {
            nt_expr_free(ptr_);
        }
    }

    // Disable copying to enforce move-only RAII semantics for expression nodes
    Expr(const Expr&) = delete;
    Expr& operator=(const Expr&) = delete;

    // Allow moving
    Expr(Expr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    Expr& operator=(Expr&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                nt_expr_free(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    static Expr var(Engine& engine, uintptr_t id, const std::string& name, DataType dtype, uint32_t bits, const std::vector<double>& values, uint32_t scale = 1) {
        ::Expr* ptr = nt_expr_new_var(engine.get(), id, name.c_str(), static_cast<uint32_t>(dtype), bits, values.data(), values.size(), scale);
        if (!ptr) {
            throw std::runtime_error("Failed to create variable expression");
        }
        return Expr(ptr);
    }

    static Expr constant(double val) {
        ::Expr* ptr = nt_expr_new_const(val);
        if (!ptr) {
            throw std::runtime_error("Failed to create constant expression");
        }
        return Expr(ptr);
    }

    Expr operator+(const Expr& other) const {
        ::Expr* ptr = nt_expr_add(ptr_, other.ptr_);
        if (!ptr) {
            throw std::runtime_error("Failed to perform addition");
        }
        return Expr(ptr);
    }

    Expr operator*(const Expr& other) const {
        ::Expr* ptr = nt_expr_mul(ptr_, other.ptr_);
        if (!ptr) {
            throw std::runtime_error("Failed to perform multiplication");
        }
        return Expr(ptr);
    }

    Expr execute(Engine& engine, const std::string& device = "cpu") {
        ::Expr* result_ptr = nt_expr_execute(engine.get(), ptr_, device.c_str());
        if (!result_ptr) {
            throw std::runtime_error("Failed to execute expression graph");
        }
        return Expr(result_ptr);
    }

    std::vector<double> unpack(Engine& engine, uintptr_t size) {
        std::vector<double> results(size);
        uintptr_t unpacked = nt_expr_unpack(engine.get(), ptr_, results.data(), size);
        results.resize(unpacked);
        return results;
    }

    ::Expr* get() const { return ptr_; }

private:
    ::Expr* ptr_ = nullptr;
};

} // namespace numtoy
