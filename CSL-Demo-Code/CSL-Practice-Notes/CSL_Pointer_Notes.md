# CSL Pointer 使用注意事项

## 1. 两种指针类型

| 类型 | 含义 | 产生方式 |
|------|------|----------|
| `*T` | 单元素指针，指向一个 `T` 值 | `&array[index]` |
| `[*]T` | 多元素指针，指向一片连续的 `T` 内存 | `&array`（取整个数组地址） |

## 2. 隐式转换规则

- ✅ `&array` → `[*]T`：取整个数组的地址可以隐式转换为多元素指针
- ❌ `&array[i]` → `[*]T`：取单个元素的地址返回 `*T`，**不能**隐式转换为 `[*]T`，需要用 `@ptrcast`

```csl
var buf: [100]f16 = @zeros([100]f16);
var ptr_ok:  [*]f16 = &buf;                        // ✅ 隐式转换
var ptr_err: [*]f16 = &buf[50];                     // ❌ expected [*]f16, got *f16
var ptr_fix: [*]f16 = @ptrcast([*]f16, &buf[50]);   // ❌ comptime pointer cast is not supported
```

## 3. `@ptrcast` 不能在 comptime 使用

CSL 全局变量初始化器在编译时（comptime）求值，而 `@ptrcast` 不支持 comptime。因此需要将偏移指针的初始化推迟到运行时：

```csl
// 全局声明：先用数组头部地址占位
var ptr: [*]f16 = &buf;

// 运行时：在 init 函数中设置真正的偏移
fn init_task() void {
    ptr = @ptrcast([*]f16, &buf[50]);  // ✅ 运行时 @ptrcast
}
```

## 4. 实践建议

- 指向**数组起始位置**：直接用 `&array`，可隐式转为 `[*]T`
- 指向**数组中间偏移**：必须用 `@ptrcast`，且只能在**函数体内（运行时）**使用
- 建议在 `init_task()` 等初始化函数中统一完成偏移指针的赋值
