# Note

## 629. K 个逆序对数组

[题目链接](https://leetcode.cn/problems/k-inverse-pairs-array/)

```
对于一个整数数组 nums，逆序对是一对满足 0 <= i < j < nums.length 且 nums[i] > nums[j]的整数对 [i, j] 。

给你两个整数 n 和 k，找出所有包含从 1 到 n 的数字，且恰好拥有 k 个 逆序对 的不同的数组的个数。由于答案可能很大，只需要返回对 109 + 7 取余的结果。

1 <= n <= 1000; 0 <= k <= 1000
```

使用动态规划完成题目，考虑 `dp[i][j]`，表示由 `1...i` 的所有排列中，恰好有 `j` 个逆序对的方案数量。将这一问题拆解成子问题：将整数 `i` 插入到原本由 `1...i-1` 组成的序列中。

考虑 `i` 的插入位置 `k(1 <= k <= i)`，由于 `i` **比之前序列中的任何一个数都大**，位于 `k` 位置的 `i` 会与之前位于 `k...i-1` 的 `i-k` 个数构成逆序对，而不会影响之前序列的排序，即相对于原序列，插入 `i` 后的序列增加了 `i-k` 个逆序对。

我们希望插入 `i` 后共有 `j` 个逆序对，则在插入前需要有且只有 `j - (i - k)` 个逆序对，这样的方案共有 `dp[i-1][j-i+k]` 种。由于 `k` 在 `[1, i]` 范围内任取，这样的方案共有 $dp[i][j] = \sum_{k=1}^{i}{dp[i-1][j-i+k]}=\sum_{k=0}^{i-1}{dp[i-1][j-k]}$ 种。

简化上式，考虑从 `dp[i][j-1]` 出发获得 `dp[i][j]`，有下式：

$$
dp[i][j] = dp[i][j-1] - dp[i-1][j-i] + dp[i-1][j]
$$

可以看到 `dp[i][j]` 可以仅从“上一行”`dp[i-1]`以及同一行的前面若干列的数据获得，使用两行数组 `pre, cur` 交换即可完成计算。动态规划的初始条件：`dp[0][0]=1` 因为这种情况下只有空集且空集含有 `0` 个逆序对；`dp` 中索引值为负数的均记为 `0`，因为不存在这样的方案。

```cpp
int kInversePairs(int n, int k) {
    int a[k+5];
    int b[k+5];
    int *pre = a;
    int *cur = b;
    int mod = 1000000007;
    memset(a, 0, sizeof(int) * (k+5));
    memset(b, 0, sizeof(int) * (k+5));
    a[0] = 1;
    for(int i = 1; i <= n; ++i) {
        cur[0] = pre[0];
        for(int j = 1; j <= k; ++j) {
            int t = cur[j-1];
            if(i && j >= i) t -= pre[j-i];
            t += pre[j];
            if(t >= mod) t -= mod;
            else if(t < 0) t += mod;
            cur[j] = t;
        }
        int *t = pre;
        pre = cur; cur = t;
    }
    return pre[k];
}
```

## 2176. 统计数组中相等且可以被整除的数对

[题目链接](https://leetcode.cn/problems/count-equal-and-divisible-pairs-in-an-array)

```
给你一个下标从 0 开始长度为 n 的整数数组 nums 和一个整数 k ，请你返回满足 0 <= i < j < n ，nums[i] == nums[j] 且 (i * j) 能被 k 整除的数对 (i, j) 的 数目 。
```

`(i*j)%k == 0` 可以拆分来看：若 `j` 能够整除 `k / gcd(i, k)`，`i*j` 就可以整除 `k`（在运算过程中，`i` 提供了因子 `gcd(i, k)`，`j` 提供了因子 `k / gcd(i, k)`）。

考虑使用筛法，`j` 范围为 `[0, n-1]`，将下标视为因子，需要记录下来每个下标及这个下标整数倍位置上的数（hashmap 记录数与数量，用于之后寻找 `k / gcd(i, k)` 的整数倍下标对应数）。

需要注意的是：对于下标 `0` 的数，只要满足 `nums[i] == nums[0]` 即可算入结果中；若遍历的 `i` 本身就是 `j = k / gcd(i, k)` 的整数倍，则会多记录一个（题目要求 `i != j`），需要剔除。最后由于每个数据都被记录成有序对 `(i, j), (j, i)` 两个，返回时需要除以 `2`。

```cpp
int countPairs(vector<int>& nums, int k) {
    int t = nums.size() > k+1 ? nums.size() : k+1;
    std::vector<std::unordered_map<int, int>> a(t);
    int s = nums.size();
    for(int i = 1; i < s; ++i) {
        for(int f = i; f < s; f += i) {
            ++a[i][nums[f]];
        }
    }

    int res = 0;
    for(int i = 0; i < s; ++i) {
        int nn = k / gcd(i, k);
        res += (nums[i] == nums[0]) + a[nn][nums[i]] - (i % nn == 0);
    }
    return res/2;
}
```

## 2707. 字符串中的额外字符

[题目链接](https://leetcode.cn/problems/extra-characters-in-a-string)

```
给你一个下标从 0 开始的字符串 s 和一个单词字典 dictionary 。你需要将 s 分割成若干个 互不重叠 的子字符串，每个子字符串都在 dictionary 中出现过。s 中可能会有一些 额外的字符 不在任何子字符串中。

请你采取最优策略分割 s ，使剩下的字符最少。
```

使用动态规划完成题目，`dp[i]` 记录子串 `s[0:i-1]` 中的最小额外字符数。对于新插入的字符 `s[i-1]`，它可能不能与之前的 `s[k:i-2]` 构成一个在字典中的字符串，此时额外字符数为 `dp[i-1]+1`；也可能 `s[j:i-1]` 构成了字典中的字符串，其中 `0 <= j < i`，此时额外字符数为 `dp[j]`。题目需要求的就是上述中最小的额外字符数，可以使用 Hash 存储 `dictionary` 方便查找子串是否在字典中。

```cpp
int minExtraChar(string s, vector<string>& dictionary) {
    std::unordered_set<string> st;
    for(auto &d: dictionary) st.insert(d);
    vector<int> dp(s.size()+1, INT_MAX);
    dp[0] = 0;
    for(size_t i = 1; i <= s.size(); ++i) {
        int min = dp[i-1] + 1;
        for(int j = 0; j < i; ++j) {
            if(st.find(s.substr(j, i-j)) != st.end()) {
                min = min < dp[j] ? min : dp[j];
            }
        }
        dp[i] = min;
    }
    return dp[s.size()];
}
```

## 1447. 最简分数

[题目链接](https://leetcode.cn/problems/simplified-fractions)

```
给你一个整数 n ，请你返回所有 0 到 1 之间（不包括 0 和 1）满足分母小于等于  n 的 最简 分数 。分数可以以 任意 顺序返回。
```

题目实际要求寻找 `(i, j), 1 <= j < i <= n` 使得 `i` 与 `j` 互质。

使用埃筛，预计算 `[1, n]` 中所有的素数，复杂度为 `O(nloglogn)`。当分母为质数时，分子可以为 `[1, n)` 的任何整数；当分母为合数时，可以遍历 `[1, i)` 寻找与分母互质的分子，也可以类似筛法，从 `[1, i)` 的素数中寻找可以整除分母的素数，将它与其整数倍标记为与分母不互质。

```cpp
vector<string> simplifiedFractions(int n) {
    vector<int> prime;
    bool isprime[n+5]; memset(isprime, true, sizeof(isprime));
    isprime[0] = false; isprime[1] = false;
    int sqrt_n = sqrt(n);
    for(int i = 2; i <= n; ++i) {
        if(isprime[i]) {
            for(int j = i*i; j <= n; j += i) isprime[j] = false;
            prime.push_back(i);
        }
    }

    int tmp[n+5];
    vector<string> res;
    for(int i = 2; i <= n; ++i) {
        if(isprime[i]) {
            for(int j = 1; j < i; ++j) {
                res.push_back(to_string(j) + '/' + to_string(i));
            }
        } else {
            memset(tmp, 1, sizeof(tmp));
            for(const int &p: prime) {
                if(p > i/2) break;
                if(i % p == 0) {
                    for(int j = p; j <= n; j += p) {
                        tmp[j] = 0;
                    }
                }
            }
            for(int j = 1; j < i; ++j) {
                if(tmp[j]) {
                    res.push_back(to_string(j) + '/' + to_string(i));
                }
            }
        }
    }
    return res;
}
```