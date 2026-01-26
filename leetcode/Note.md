# Note

## 629. K 个逆序对数组

[题目链接](https://leetcode.cn/problems/k-inverse-pairs-array/)

**Keywords:** 动态规划

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

**Keywords:** 数组技巧

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

**Keywords:** 动态规划；字典树

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

**Keywords:** 埃筛

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

## 782. 变为棋盘

[题目链接](https://leetcode.cn/problems/transform-to-chessboard/)

**Keywords:** 位运算；矩阵

```
一个 n x n 的二维网络 board 仅由 0 和 1 组成 。每次移动，你能任意交换两列或是两行的位置。

返回 将这个矩阵变为  “棋盘”  所需的最小移动次数 。如果不存在可行的变换，输出 -1。

“棋盘” 是指任意一格的上下左右四个方向的值均与本身不同的矩阵。
```

思考量不大，但是因为每个判断与处理都要涉及行和列，代码量较大。观察棋盘，先考虑哪些给定网格不能组成棋盘，对于一行，由于换列时，不同行整列同时操作，要实现棋盘形，那么每一行要么与第一行**完全相同**，要么与第一行**完全相反**，且与第一行相同的行数（包括第一行）与相反的行数至多差 `1`（`n` 为奇数时差一，为偶数时相同）；对列的要求几乎相同，不再赘述。

在判断能够构成棋盘后，要分行移动与列移动来统计移动数量。总的思路是确定**原第一行仍位于第一行**还是**与原第一行相反的行应位于第一行**，这也是这道题目拖时间比较久的原因。

```cpp
int movesToChessboard(vector<vector<int>>& board) {
    size_t n = board.size();
    int cnt_same0_row = 1;
    vector<bool> is_same0(n, true);
    vector<int> &r0 = board[0];
    for(size_t i = 1; i < n; ++i) {
        vector<int> &rt = board[i];
        if(r0[0] == rt[0]) {
            for(int j = 1; j < n; ++j) if(r0[j] != rt[j]) return -1;
            ++cnt_same0_row;
        } else {
            for(int j = 1; j < n; ++j) if(r0[j] == rt[j]) return -1;
            is_same0[i] = false;
        }
    }
    if(((n%2 == 0) && cnt_same0_row != n-cnt_same0_row) || ((n%2==1) && (2*cnt_same0_row+1 != n && 2*cnt_same0_row-1 != n))) return -1;
    
    // check col
    int cnt_same0_col = 1;
    vector<bool> is_same0_col(n, true);
    for(int j = 1; j < n; ++j) {
        if(board[0][0] == board[0][j]) {
            for(int i = 1; i < n; ++i) if(board[i][j] != board[i][0]) return -1;
            ++cnt_same0_col;
        } else {
            for(int i = 1; i < n; ++i) if(board[i][j] == board[i][0]) return -1;
            is_same0_col[j] = false;
        }
    }
    if(((n%2 == 0) && cnt_same0_col != n-cnt_same0_col) || ((n%2==1) && ((2*cnt_same0_col+1 != n) &&(2*cnt_same0_col-1 != n)))) return -1;

    int move_row = 0;
    int cnt_same0_even = 0;
    for(int i = 0; i < n; i += 2) {
        cnt_same0_even += is_same0[i];
    }
    if(n%2 == 0) {
        if(cnt_same0_even >= n/2) {
            move_row = cnt_same0_even - n/2;
        } else {
            move_row = n/2 - cnt_same0_even;
            move_row = move_row > cnt_same0_even ? cnt_same0_even : move_row;
        }
    } else {
        if(cnt_same0_row * 2 > n) {
            move_row = (n+1)/2-cnt_same0_even;
        } else {
            move_row = cnt_same0_even;
        }
    }

    int move_col = 0;
    cnt_same0_even = 0;
    for(int i = 0; i < n; i += 2) {
        cnt_same0_even += is_same0_col[i];
    }
    if(n%2 == 0) {
        if(cnt_same0_even >= n/2) {
            move_col = cnt_same0_even - n/2;
        } else {
            move_col = n/2 - cnt_same0_even;
            move_col = move_col > cnt_same0_even ? cnt_same0_even : move_col;
        }
    } else {
        if(cnt_same0_col * 2 > n) {
            move_col = (n+1)/2-cnt_same0_even;
        } else {
            move_col = cnt_same0_even;
        }
    }

    return move_col + move_row;
}
```

## 2935. 找出强数对的最大异或值 II

[题目链接](https://leetcode.cn/problems/maximum-strong-pair-xor-ii)

**Keywords:** 哈希；位运算

完全没有做出来的题目，最开始的想法是从低位开始遍历，寻找尽可能多的位不同的点，但找不到能将复杂度降到 `O(N^2)` 的方法。参考[题解](https://leetcode.cn/problems/maximum-strong-pair-xor-ii/solutions/2523213/0-1-trie-hua-dong-chuang-kou-pythonjavac-gvv2/)才做出来。

思路与刚才提到的部分一致，但是应该从高位开始，依次尝试最终答案 `res` 能否为 `1`。尝试的过程就是根据当前的 `res`（低位均为 `0`）与数组进行异或，得到的结果在一个 Hashmap 中查找，这个 Hashmap 记录了掩码后的值（使低位为 `0`）与本身的映射。在每个数值的位宽为常数时，这个算法的复杂度为 `O(NlogN)`。

```cpp
int maximumStrongPairXor(vector<int>& nums) {
    // sort
    sort(nums.begin(), nums.end());
    
    // for x < y; y - x <= x ---> y <= 2*x ---> y <= x << 1
    unsigned int res = 0;
    size_t sz = nums.size();
    unsigned int mask = 0;
    for(int i = 31; i >= 0; --i) {
        unsigned int new_res = res | ((unsigned int)1 << i);
        mask |= 1 << i;
        unordered_map<int, int> m;
        for(const int &a: nums) {
            unsigned int mask_a = mask & a;
            auto it = m.find(mask_a ^ new_res);
            if(it != m.end() && it->second << 1 >= a) {
                res = new_res;
                break;
            }
            m[mask_a] = a;
        }
    }
    return res;
}
```

这个题目还学到了一些东西，比如 gcc 提供的[内置函数](https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/Other-Builtins.html) `__builtin_clz, builtin_ctz` 等等，方便对二进制位的统计和操作。题目还可以通过[字典树](https://oi-wiki.org/string/trie/)来完成，进行了初步的学习但没有尝试编写代码。

## 481. 神奇字符串

[题目链接](https://leetcode.cn/problems/magical-string/)

**Keywords:** 双指针

```
神奇字符串 s 仅由 '1' 和 '2' 组成，并需要遵守下面的规则：

神奇字符串 s 的神奇之处在于，串联字符串中 '1' 和 '2' 的连续出现次数可以生成该字符串。
s 的前几个元素是 s = "1221121221221121122……" 。如果将 s 中连续的若干 1 和 2 进行分组，可以得到 "1 22 11 2 1 22 1 22 11 2 11 22 ......" 。每组中 1 或者 2 的出现次数分别是 "1 2 2 1 1 2 1 2 2 1 2 2 ......" 。上面的出现次数正是 s 自身。

给你一个整数 n ，返回在神奇字符串 s 的前 n 个数字中 1 的数目。
```

使用快慢指针，慢指针每次走 `1` 步，快指针每次走 `a[slow]` 步，其中 `a` 是一个容量 `n+2` 的数组。思考很久有没有空间 `O(1)` 的方法但无果，主要是因为要记录快指针走过的位置（用 `t` 表示快指针路过时应写入的值 `1` 或 `2`）。`t` 在每次快指针走动后需要反转，并写入快指针的“落脚点”。需要注意的是，因为快指针可能走到 `n` 外，为了方便代码书写，数组申请了 `n+2` 个空间，并在最后判断数组的多余两个元素是否有 `1`，如有需要减去对应的数量，再返回结果。

使用快慢指针的本质是，快指针指向第一层数组，慢指针指向第二层数组。

```cpp
int magicalString(int n) {
    int res = 0;
    int t = 1;
    vector<int> a(n+2, 0);
    int fast = 0;
    int slow = 0;

    a[0] = 1;
    res = 1;
    while(fast < n) {
        // decide how many hops does fast take
        int hops = a[slow];
        if(hops == 1) {
            fast++;
            t = t == 1 ? 2 : 1;
            if(t == 1) ++res;
        } else {
            // hops = 2
            a[fast+1] = t;
            t = t == 1 ? 2 : 1;
            fast += 2;
            ++res;
        }
        a[fast] = t;
        slow++;
    }
    if(a[n+1] == 1) --res;
    if(a[n] == 1) --res;

    return res;
}
```

## 910. 最小差值 II

[题目链接](https://leetcode.cn/problems/smallest-range-ii/)

**Keywords:** 贪心

```
给你一个整数数组 nums，和一个整数 k 。

对于每个下标 i（0 <= i < nums.length），将 nums[i] 变成 nums[i] + k 或 nums[i] - k 。

nums 的 分数 是 nums 中最大元素和最小元素的差值。

在更改每个下标对应的值之后，返回 nums 的最小 分数 。
```

（将数组排序后）用贪心策略，把小数变大，把大数变小。寻找一个下标 `idx` 使 `[0, idx)` 的元素变大，使 `[idx, N-1]` 的元素变小，寻找这样变化条件下的最大值与最小值之差。此时最大值是 `nums[idx-1]+k` 与 `origin_max-k` 中较大的一个；最小值是 `nums[idx]-k` 与 `origin_min+k` 中较小的一个；上述两者的差值就是选取 `idx` 作为增大/减少分界时的最值之差，从中寻找最小的最值之差即为本题目的答案。

```cpp
int smallestRangeII(vector<int>& nums, int k) {
    sort(nums.begin(), nums.end());
    int ma = nums.back();
    int mi = nums[0];
    int res = ma - mi;
    for(int i = 1; i < nums.size(); ++i) {
        int upper = nums[i-1] + k > ma-k ? nums[i-1]+k : ma-k;
        int lower = nums[i]-k < mi+k ? nums[i]-k : mi+k;
        res = res > upper-lower ? upper-lower : res;
    }
    return res;
}
```

## 1780. 判断一个数字是否可以表示成三的幂的和

[题目链接](https://leetcode.cn/problems/check-if-number-is-a-sum-of-powers-of-three/)

**Keywords:** 数学

```
给你一个整数 n ，如果你可以将 n 表示成若干个不同的三的幂之和，请你返回 true ，否则请返回 false 。

对于一个整数 y ，如果存在整数 x 满足 y == 3x ，我们称这个整数 y 是三的幂。
```

从“三进制”角度出发，每一个 3 的幂的系数不是 `1` 就是 `0`，不能出现 `2`；从竖式角度出发，就是每一步的余数不能是 `2`。

```cpp
bool checkPowersOfThree(int n) {
    while(n > 0) {
        int rem = n % 3;
        if(n % 3 == 2) return false;
        n /= 3;
    }
    return true;
}
```

## 1278. 分割回文串 III

[题目链接](https://leetcode.cn/problems/palindrome-partitioning-iii/)

**Keywords:** 动态规划

```
给你一个由小写字母组成的字符串 s，和一个整数 k。

请你按下面的要求分割字符串：

首先，你可以将 s 中的部分字符修改为其他的小写英文字母。
接着，你需要把 s 分割成 k 个非空且不相交的子串，并且每个子串都是回文串。
请返回以这种方式分割字符串所需修改的最少字符数。
```

用动态规划解决问题。`dp[i][j]` 表示将前 `i` 个字符构成的字符串划分为 `j` 个回文串最少改变的字符数量。基于已有 `j-1` 个回文串的子问题，需要考虑添加的包含 `s[i]` 的第 `j` 个字符串，需要枚举这个字符串开始的下标 `i0`，其中 `0 <= i0 <= i`。有 `dp[i][j] = min(dp[i0][j-1] + cost(i0+1, i))`，其中 `cost(i, j)` 表示将 `s[i:j]` 子串变为回文串需要改变的字符数量（可以通过指向首尾的双指针统计获得，但会导致时间复杂度过高）。

为了降低 `cost` 的计算成本，需要预处理得到 `cost[i][j]`，这一步也可以使用动态规划来完成。对于 `i >= j` 的情况，对应空字符串，`cost=0`；其他符合边界的条件下 `cost[i][j] = cost[i+1][j-1] + (s[i] != s[j])`。观察递推关系，可以从大到小遍历 `i`，从小到大遍历 `j` 来完成。

```cpp
int palindromePartition(string s, int k) {
    int len = s.length();
    int c[len+1][len+1];
    memset(&c[len-1], 0, sizeof(int) * (len+1));
    for(int i = len-2; i>= 0; --i) {
        c[i][0] = 0;
        for(int j = 1; j < len; ++j) {
            if(i >= j) c[i][j] = 0;
            else c[i][j] = c[i+1][j-1] + (s[i] != s[j]);
        }
    }
    int dp[len+1][k+1];
    dp[0][0] = 0;
    for(int i = 1; i <= len; ++i) {
        for(int j = 1; j <= k && j <= i; ++j) {
            if(j == 1) {
                dp[i][j] = c[0][i-1];
            } else {
            int min = INT_MAX;
            for(int i0 = j-1; i0 < i; ++i0) {
                int t = dp[i0][j-1] + c[i0][i-1];
                min = min < t ? min : t;
            }
            dp[i][j] = min;
        }
        }
    }
    return dp[len][k];
}
```

## 115. 不同的子序列

[题目链接](https://leetcode.cn/problems/distinct-subsequences/)

**Keywords:** 动态规划

```
给你两个字符串 s 和 t ，统计并返回在 s 的 子序列 中 t 出现的个数，结果需要对 109 + 7 取模。
```

考虑使用动态规划解决问题。使用 `dp[i][j]` 来表示 `s[i:]` 的子序列中中 `t[j:]` 出现的个数。若 `s[i]==t[j]`，则可以分为两位匹配与不匹配两种情况，对于两位匹配的情况，匹配数量与 `dp[i+1][j+1]` 相同（这一位匹配，只需要找子串后边的数量），对于两位不匹配的情况，匹配数量与 `dp[i+1][j]` 相等（即 `s[i+1:` 与 `t[j:]` 的匹配情况；若 `s[i]!=t[j]` 则只有不匹配的情况，此时匹配数量为 `dp[i+1][j]`。总结如下：

```
dp[i][j] = (s[i]==s[j]) ? dp[i+1][j+1]+dp[i+1][j] : dp[i+1][j]
```

观察到这个递推关系在第 `i` 行仅与 `i+1` 行有关系，则可以重复利用一行 `dp[]` 数组存放。即：

```
dp_new[j] = (s[i]==s[j]) ? dp_old[j+1]+dp_old[j] : dp_old[j]
```

考察边界条件 `dp[len_s][len_t]` 表示两个空串的子序列匹配，则其值 `dp[len_t]=1`。代码中需要注意的细节：`i` 需要从大到小遍历，以在 `i` 行利用 `i+1` 行的结果；在使用一行记录的情况下，`j` 需要从小到大遍历，以避免覆盖 `dp[j+1]` 的值。

```cpp
int numDistinct(string s, string t) {
    int mod = 1e9 + 7;
    size_t len_s = s.length();
    size_t len_t = t.length();
    if(len_s < len_t) return 0;

    vector<int> dp(1005, 0);
    dp[len_t] = 1;
    for(int i = len_s-1; i >= 0; --i) {
        for(int j = 0; j < len_t; ++j) {
            if(s[i] == t[j]) {
                dp[j] = dp[j+1] + dp[j];
                dp[j] %= mod;
            } else {
                dp[j] = dp[j];
                dp[j] %= mod;
            }
        }
    }
    return dp[0];
}
```

## 2958. 最多 K 个重复元素的最长子数组

[题目链接](https://leetcode.cn/problems/length-of-longest-subarray-with-at-most-k-frequency/)

**Keywords:** 哈希；滑动窗口

```
给你一个整数数组 nums 和一个整数 k 。

一个元素 x 在数组中的 频率 指的是它在数组中的出现次数。

如果一个数组中所有元素的频率都 小于等于 k ，那么我们称这个数组是 好 数组。

请你返回 nums 中 最长好 子数组的长度。

子数组 指的是一个数组中一段连续非空的元素序列。
```

使用滑动窗口解决问题，利用一个 Hashmap 记录现在窗口中每个数有多少个。当遇到窗口右侧扩展时，对应 `nums[right]` 在窗口中已经有 `k` 个时，需要通过增加 `left` 来缩短窗口，并对应地更新 Hashmap。

```cpp
int maxSubarrayLength(vector<int>& nums, int k) {
    if(k == 0) return 0;
    int res = 0;
    int left = 0;
    unordered_map<int, int> m;
    for(size_t i = 0; i < nums.size(); ++i) {
        int t = nums[i];
        if(m[t] == k) {
            // need to move 'left'
            if(i-left > res) res = i-left;
            while(nums[left] != t) {
                m[nums[left]]--;
                ++left;
            }
            ++left;
        } else {
            m[t]++;
        }
    }
    if(nums.size()-left > res) res = nums.size()-left;
    return res;
}
```

## 2850. 将石头分散到网格图的最少移动次数

[题目链接](https://leetcode.cn/problems/minimum-moves-to-spread-stones-over-grid/)

**Keywords:** 广度优先搜索

```
给你一个大小为 3 * 3 ，下标从 0 开始的二维整数矩阵 grid ，分别表示每一个格子里石头的数目。网格图中总共恰好有 9 个石头，一个格子里可能会有 多个 石头。

每一次操作中，你可以将一个石头从它当前所在格子移动到一个至少有一条公共边的相邻格子。

请你返回每个格子恰好有一个石头的 最少移动次数 。
```

只能想到用全排列一个个尝试，首先有一个来源数组 `from` 存放所有有多个（不止一个）石头的格子坐标，以及一个去向数组 `to` 来存放所有没有石头的格子坐标。[题解](https://leetcode.cn/problems/minimum-moves-to-spread-stones-over-grid/solutions/2435313/tong-yong-zuo-fa-zui-xiao-fei-yong-zui-d-iuw8/)中使用了 [next_permutation](https://en.cppreference.com/w/cpp/algorithm/next_permutation) 可以获得字典序的下一个排列（按照 `operator <` 进行比较），当传入的排列不是字典序最大时返回 `true` 并将迭代器之间的内容更新为下次排列；当传入的排列是字典序最大时返回 `false` 并将迭代器之间的内容更新成字典序最小的排列。

```cpp
int minimumMoves(vector<vector<int>>& grid) {
		vector<pair<int, int>> from;
		vector<pair<int, int>> to;
		for(int i = 0; i < grid.size(); ++i) {
				for(int j = 0; j < grid[0].size(); ++j) {
						if(grid[i][j] == 0) {
								to.push_back({i, j});
						} else if(grid[i][j] > 1) {
								for(int _ = 0; _ < grid[i][j]-1; ++_) from.push_back({i, j});
						}
				}
		}

		int res = INT_MAX;
		do{
				int tot = 0;
				for(int i = 0; i < from.size(); ++i) {
						tot += abs(from[i].first - to[i].first) + abs(from[i].second - to[i].second);
				}
				res = min(res, tot);
		} while(next_permutation(from.begin(), from.end()));

		return res;
}
```

代码中需要注意以下几点：

- 仅对 `from` 进行了全排列尝试，因为题目保证 `from` 与 `to` 元素数量相同（石头刚好有 9 个），因此 `to` 顺序不动，对 `from` 进行全排列，即可获得所有从 `from` 到 `to` 的一一映射。
- 没有任何对 `from` 的排序操作，因为我们在遍历 `grid` 的时候分别让行和列从 `0` 自增，因此获得的 `from` 就是字典序最小的一个排列。

很显然全排列的代价很大，时间复杂度能到 `O(mn * (mn)!)`，题解提到了[最小费用最大流](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E8%B4%B9%E7%94%A8%E6%9C%80%E5%A4%A7%E6%B5%81%E9%97%AE%E9%A2%98)，但没有看得太明白 orz

## 968. 监控二叉树

[题目链接](https://leetcode.cn/problems/binary-tree-cameras/)

**Keywords:** 树形动态规划；动态规划

```
给定一个二叉树，我们在树的节点上安装摄像头。

节点上的每个摄影头都可以监视其父对象、自身及其直接子对象。

计算监控树的所有节点所需的最小摄像头数量。
```

参考了[题解](https://leetcode.cn/problems/binary-tree-cameras/solutions/2452795/shi-pin-ru-he-si-kao-shu-xing-dpgai-chen-uqsf/)，使用树形 DP 完成题目，思路是使用 DFS 从下到上传递信息。每一个节点需要保证自己能被监控到，有三种情况下可以做到：自己是监控(`choose_this`)、自己的父节点是监控(`by_father`)或者自己的子节点是监控(`by_child`)。对于每一个节点，需要先获得其子节点的信息。

对于当前节点

- *如果选择当前节点作为监控*，左右子节点并不需要它的子节点作为监控，比较 `by_father` 与 `choose` 即可（这里用了贪心，子节点（1 个）作为监控时不需要考虑子节点的子节点（可能 2 个），因此在比较时没有考虑 `by_child`）；
- *如果父节点作为监控*，则子节点需要保证子节点自己被监控到，从 `choose` 与 `by_child` 中寻找最小值，而当前节点一定保证可见，不需要添加监控点；
- *如果父节点与自己都不是监控*，需要子节点保证当前节点被监控到，即至少有一个子节点是监控。

代码的细节：对于 NULL 节点的处理，`choose_this` 设置为了 `INT_MAX/2` 为了规避可能产生的两数相加的溢出。

```cpp
void dfs(TreeNode *t, int *choose_this, int *by_father, int *by_child) {
		if(t == NULL) {
				*choose_this = INT_MAX >> 1;
				*by_father = 0;
				*by_child = 0;
				return;
		}
		int l_choose, l_by_father, l_by_child;
		int r_choose, r_by_father, r_by_child;
		dfs(t->left, &l_choose, &l_by_father, &l_by_child);
		dfs(t->right, &r_choose, &r_by_father, &r_by_child);
		*choose_this = min(l_by_father, l_choose) + min(r_by_father, r_choose) + 1;
		*by_father = min(l_choose, l_by_child) + min(r_choose, r_by_child);
		*by_child = min(min(l_choose+r_by_child, l_by_child+r_choose), l_choose+r_choose);
}
int minCameraCover(TreeNode* root) {
		int choose_this, by_father, by_child;
		dfs(root, &choose_this, &by_father, &by_child);
		return min(choose_this, by_child);
}
```

## 1155. 掷骰子等于目标和的方法数

[题目链接](https://leetcode.cn/problems/number-of-dice-rolls-with-target-sum/description/)

**Keywords:** 动态规划

```
这里有 n 个一样的骰子，每个骰子上都有 k 个面，分别标号为 1 到 k 。

给定三个整数 n、k 和 target，请返回投掷骰子的所有可能得到的结果（共有 kn 种方式），使得骰子面朝上的数字总和等于 target。

由于答案可能很大，你需要对 109 + 7 取模。
```

比较明显是一个动态规划问题，为了求 `dp[i][j]`（其含义为有 `i` 个骰子和为 `j` 的情况总个数）可以使用 `dp[i-1][j-k...j-1]` 的总和计算得到：即增加一个骰子其对总和的影响是 `1...k`，几个特殊条件也比较好思考：

- 当 `i==0` 时有 `dp[0][j] = (j == 0) ? 1 : 0`
- 当 `j==0` 时有 `dp[i][0] = (i == 0) ? 1 : 0`

代码如下，其实还能更简化，我只需要持有一个 `sum` 以及一个 `arr[k]` 即可，使用 `sum` 进行窗口内总和的统计，使用 `arr[k]` 判断即将出窗口的数量是多少，但是这样会增加代码的复杂度，对于算法题来说意义不大，故没有完成。

```cpp
#define MOD_NUM (1000000000 + 7)
class Solution {
public:
    int sum_of_pre_k(std::vector<int>& arr, int j, int k) {
      int sum = 0;
      for(int i = j-k; i < j; ++i) {
        if(i >= 0) sum += arr[i];
        sum %= MOD_NUM;
      }
      return sum;
    }
    int numRollsToTarget(int n, int k, int target) {
      if(target <= 0 || n <= 0 || k <= 0) return 0;
      std::vector<int> arr(target + 100, 0);
      std::vector<int> arr_new(target+100, 0);
      arr[0] = 1;
      for(int i = 1; i <= n; ++i) {
        for(int j = 1; j <= target; ++j) {
          arr_new[j] = sum_of_pre_k(arr, j, k);
        }
        arr = std::move(arr_new);
        arr_new.assign(target+100, 0);
      }

      return arr[target];
    }
};
```

## 1995. 统计特殊四元组

[题目链接](https://leetcode.cn/problems/count-special-quadruplets/description/)

**Keywords:** 哈希；构造

```
给你一个 下标从 0 开始 的整数数组 nums ，返回满足下述条件的 不同 四元组 (a, b, c, d) 的 数目 ：

- nums[a] + nums[b] + nums[c] == nums[d] ，且

- a < b < c < d
```

主要限制点在于 abcd 下标各不相同且递增，不能通过排序/建立 mapping 的方法压缩加/减的查询数量，即题目几乎只能用遍历的方式来找，主要的目标是减少遍历维度，进而压缩复杂度。

注意到要求的限制也可以写为 `nums[a] + nums[b] == nums[d] - nums[c]`，且 `a<b<c<d` 那么其实可以 `b` 为轴，进行遍历（`b \in [0, n-3]`），并用 hash 记录大于 `b` 的 `c, d` 相减的值，那么总体思路就可以写成：

- 在 `[1, n-3]` 遍历 `b`
    - 在 `[b+1, n-1]` 遍历 `c, d`，这里有一个点是逆序遍历 `b`，这样的话可以固定一端 `c`，从 `[b+2, n-1]` 遍历 `d` 即可，因为 `c > b+1` 的情况已经在之前遍历过了。获取 `counter[nums[d]-nums[c]]`
    - 在 `[0, b-1]` 遍历 `a`，只需要利用 `counter` 查询 `nums[a]+nums[b]` 的值，即可获得所有 `nums[a]+nums[b] == nums[d]-nums[c]` 的情况数了

```cpp
class Solution {
public:
    int countQuadruplets(vector<int>& nums) {
      if(nums.size() < 4) return 0; 
      int n = nums.size();
      std::unordered_map<int, int> m;
      int res = 0;
      for(int b = n-3; b >= 1; b--) {
        // build cnt map for (num[d] - num[c] for d > c > b)
        for(int d = n-1; d > b+1; --d) {
          m[nums[d] - nums[b+1]]++;
        }

        for(int a = 0; a < b; ++a) {
          if(m.count(nums[a] + nums[b])) res += m[nums[a] + nums[b]];
        }
      }
      return res;
    }
};

```
## 2560. 打家劫舍 IV

[题目链接](https://leetcode.cn/problems/house-robber-iv/description/)

**Keywords:** 二分查找；动态规划；最小化最大值

```
沿街有一排连续的房屋。每间房屋内都藏有一定的现金。现在有一位小偷计划从这些房屋中窃取现金。

由于相邻的房屋装有相互连通的防盗系统，所以小偷 不会窃取相邻的房屋 。

小偷的 窃取能力 定义为他在窃取过程中能从单间房屋中窃取的 最大金额 。

给你一个整数数组 nums 表示每间房屋存放的现金金额。形式上，从左起第 i 间房屋中放有 nums[i] 美元。

另给你一个整数 k ，表示窃贼将会窃取的 最少 房屋数。小偷总能窃取至少 k 间房屋。

返回小偷的 最小 窃取能力。
```

第一次遇到类似的题目，参考了[题解](https://leetcode.cn/problems/house-robber-iv/solutions/2093952/er-fen-da-an-dp-by-endlesscheng-m558/)才最终理解。

这是一个“最小化最大值”的题目，即对于“找到一个 k 长度的子序列，得到子序列最大值”这个问题求解最小值。对于“最大化最小值”或者“最小化最大值”的题目可以利用二分法夹逼最终的答案。

比如对于本题，可以定义一个函数 `f(i, val)` 其含义为 `nums[0...i]` 中偷不超过 `val` 价值的房屋，最多能偷几间。下面讨论 val 固定时，`f(i)` 的解法，显然可以用动态规划完成，即当不选择偷第 `i` 房间时 `f(i) = f(i-1)`；而当 `nums[i] <= val` 可以进行选择时，则 `f(i) = max(f(i-1), f(i-2)+1)`，即可选偷本房间或不偷，不偷则和前例相同，偷则可以比 `f(i-2)` 刚好多偷一间房屋。

利用上述方法，可以获得 `f(n-1, val)` 的值，这个值的含义就是可以从所有房屋中选择，偷取价值不超过 `val` 的房间，最多能偷多少间。

在获得这个值之后，可以开始考虑二分查找的方式和边界变动条件了。借用题解的一句话说，二分查找的本质是“一般地，二分的值越小，越不能/能满足要求；二分的值越大，越能/不能满足要求。有单调性的保证，就可以二分答案了。”对于本题，我们取中点值 `mid`，若 `f(n-1, mid) >= k` 则代表着最终的答案最多为 `mid`（为了获取可能的更小的值，我们需要把右边界划到 `mid`），反之则代表答案一定超过 `mid`（因为这种情况下我们只偷 `mid` 以下的房子，是偷不到目标 `k` 的，需要把左边界划到 `mid`）。

有了这些分析，已经能够写出代码了。但是对于二分查找的规范写法，之前并没有思考过，导致二分的边界时常出问题，这里也总结一下基本的二分查找写法：首先需要认识到二分查找的边界含义（对于相邻终止模板为 `left+1 < right` 的情形），`left` 始终是不满足条件的最大值（对于题目就是 `val <= left` 时，无法偷到 `k` 间），而 `right` 始终是满足条件的最小值（当 `val >= right` 时，总能偷够 `k` 间），这种情况下，进行边界收缩需要以 `left=mid or right=mid` 进行，因为我们总能判断出 `mid` 能否满足/不满足条件，而不确定 `mid-1 or mid+1` 能否满足条件。

```cpp
class Solution {
public:
    int minCapability(vector<int>& nums, int k) {
        int left = 0, right;
        right = *std::max_element(nums.begin(), nums.end());
        while(left + 1 < right) {
          int mid = (left + right) / 2;
          // check the result, f2 for f(i-2), f1 for f(i-1), f for f(i)
          int f2 = 0, f1 = 0;
          for(int i = 0; i < nums.size(); ++i) {
            if(nums[i] > mid) {
              // for next iter 
              f2 = f1;
            } else {
              int max = std::max(f1, f2+1);
              f2 = f1; f1 = max;
            }
          }
          if(f1 >= k) {
            right = mid;
          } else {
            left = mid;
          }
        }
        return right;
    }
};
```

## 576. 出界的路径数

[题目链接](https://leetcode.cn/problems/out-of-boundary-paths/description/)

**Keywords:** 动态规划
```
```

```
给你一个大小为 m x n 的网格和一个球。球的起始坐标为 [startRow, startColumn] 。你可以将球移到在四个方向上相邻的单元格内（可以穿过网格边界到达网格之外）。你 最多 可以移动 maxMove 次球。

给你五个整数 m、n、maxMove、startRow 以及 startColumn ，找出并返回可以将球移出边界的路径数量。因为答案可能非常大，返回对 109 + 7 取余 后的结果。
```

比较明显是一个动态规划问题，这个动态规划问题可以拆分为两维：一维是移动的步数，另一维是一个矩阵。综合来看，可以用一个三维 dp 来描述，对于 `dp[k][i][j]` 其含义是动 `k` 步之后，能够到达位置 `(i, j)` 的路径数量。其特殊情况是第 0 步时，即 `dp[0]`，只有开始点有一个路径（站着不动）能达到，矩阵上的其他位置都是 0，即 `dp[0][i][j] = (i == startRow && j == startColumn) ? 1 : 0`。

之后考虑 `dp[k]` 应该怎么由之前的状态 `dp[k-1]` 获得，可以简单发现，对于 `(i, j)` 来说，只有上一步在 `(i-1, j), (i+1, j), (i, j-1), (i, j+1)` 这四个位置的路径可以走一步到达。我们可以获得递推式 `dp[k][i][j] = dp[k-1][i-1][j] + dp[k-1][i+1][j] + dp[k-1][i][j-1] + dp[k-1][i][j+1]`，当然，需要注意在边界上的点可能会缺少其中的某一项或某几项，其本质是因为出界的所有路径都不会再回来，这也是对正确性的保证。

在得到 `dp` 后，我们可以将 `dp[k]` 矩阵的边界上的所有数据进行加和，得到再走一步就能出界的路径数量，需要注意，四角上的点都各有两个选择，因此应该记录两遍。

对于这个题目，得到 `dp` 以后还不能直接完成，因为我们的答案是走 `[1, maxMove]` 步能出界的路径数，而不只是走 `maxMove` 步出界的路径数。因此在每走一步之前，都需要统计下来矩阵的额边界值之和，即最终的答案是 `sum(sum_boundary) for step in [0, maxMove-1]`。

同时还有几个特别情况需要特殊处理：

- 当 `maxMove == 0` 时，答案一定是 0

- 当 `m == 1, n == 1` 时，只要 `maxMove > 0` 就有且只有 4 个路径能出界

- 当 `m == 1, n > 1` 时，边界上的两个点有三个方向可以出界，`m > 1, n == 1` 时同理

```cpp
#define MOD_NUM (1000000000+7)

class Solution {
public:
    long long int sum_mid_state(std::vector<std::vector<long long int>> &dp, int m, int n) {
      long long int sum = 0;

      if(m == 1) {
        for(int j = 1; j < n-1; ++j) sum += 2 * dp[0][j], sum %= MOD_NUM;
        sum += 3 * dp[0][0], sum %= MOD_NUM, sum += 3 * dp[0][n-1], sum %= MOD_NUM;
        return sum;
      }
      if(n == 1) {
        for(int i = 1; i < m-1; ++i) sum += 2 * dp[i][0], sum %= MOD_NUM;
        sum += 3 * dp[0][0], sum %= MOD_NUM, sum += 3 * dp[m-1][0], sum %= MOD_NUM;
        return sum;
      }

      for(int i = 1; i < m-1; ++i) sum += dp[i][0], sum %= MOD_NUM, sum += dp[i][n-1], sum %= MOD_NUM;
      for(int j = 1; j < n-1; ++j) sum += dp[0][j], sum %= MOD_NUM, sum += dp[m-1][j], sum %= MOD_NUM;
      sum += 2 * dp[0][0], sum %= MOD_NUM, sum += 2 * dp[0][n-1], sum %= MOD_NUM;
      sum += 2 * dp[m-1][0], sum %= MOD_NUM, sum += 2 * dp[m-1][n-1], sum %= MOD_NUM;

      return sum;
    }
    int findPaths(int m, int n, int maxMove, int startRow, int startColumn) {
      if(m == 1 && n == 1) return 4;
      if(maxMove <= 0) return 0;
      std::vector<std::vector<long long int>> dp_new;
      std::vector<std::vector<long long int>> dp;
      long long int res = 0;
      dp.resize(m); for(int i = 0; i < m; ++i) dp[i].assign(n, 0);
      dp_new.resize(m); for(int i = 0; i < m; ++i) dp_new[i].assign(n, 0);

      dp[startRow][startColumn] = 1;
      for(int k = 0; k < maxMove-1; ++k) {
        res += sum_mid_state(dp, m, n);
        res %= MOD_NUM;
        for(int i = 0; i < m; ++i) {
          for(int j = 0; j < n; ++j) {
            long long int sum = 0;
            if(i > 0) sum += dp[i-1][j], sum %= MOD_NUM;
            if(j > 0) sum += dp[i][j-1], sum %= MOD_NUM;
            if(i < m-1) sum += dp[i+1][j], sum %= MOD_NUM;
            if(j < n-1) sum += dp[i][j+1], sum %= MOD_NUM;
            
            dp_new[i][j] = sum;
          }
        }

        std::swap(dp_new, dp);
      }

      res += sum_mid_state(dp, m, n);
      res %= MOD_NUM;

      return res;
    }
};
```

## 1391. 检查网格中是否存在有效路径

[题目链接](https://leetcode.cn/problems/check-if-there-is-a-valid-path-in-a-grid/description/)

**Keywords:** 并查集；无向图连通性

```
给你一个 m x n 的网格 grid。网格里的每个单元都代表一条街道。grid[i][j] 的街道可以是：

1 表示连接左单元格和右单元格的街道。
2 表示连接上单元格和下单元格的街道。
3 表示连接左单元格和下单元格的街道。
4 表示连接右单元格和下单元格的街道。
5 表示连接左单元格和上单元格的街道。
6 表示连接右单元格和上单元格的街道。

你最开始从左上角的单元格 (0,0) 开始出发，网格中的「有效路径」是指从左上方的单元格 (0,0) 开始、一直到右下方的 (m-1,n-1) 结束的路径。该路径必须只沿着街道走。

注意：你 不能 变更街道。

如果网格中存在有效的路径，则返回 true，否则返回 false 。
```

本来是用 DFS 来完成的，后来查看题解发现“并查集”的使用更加简单和适用，而且对于无向图连通性的处理上，并查集在均摊复杂度上有着极大的优势。

简单描述并查集的含义，对于一个数组 `parent[n]` 其中每一个元素表示自己的父节点下标，可以通过不断做 `parent[parent[...parent[i]]]` 来索引直到一个 `x` 有 `parent[x] == x`，这个 `x` 被称为根节点，路径上的所有点（`i, parent[i], parent[parent[i]]...`）都属于这个根，而这个根节点与所有子节点的集合表示一个连通量（即集合内的所有点都能经过有限步到达任意另一个点）。

并查集的主要操作是 `find` 以及 `union`，前者通过迭代来找到根节点，后者将两个根节点进行合并，表示两个连通量进行合并。具体优化上，可以在 `find` 过程中将所有节点对应的 `parent` 值直接修改为 `root` 来减少层数，节约查询代价，以此实现均摊 `O(1)` 的查询代价。而 `union` 只需要负责找到两个点的根节点，并将其中一个的 `parent` 修改即可。

在完成并查集的实现后，本题就变成纯粹的体力劳动了。

