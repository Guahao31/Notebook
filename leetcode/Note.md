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

## 782. 变为棋盘

[题目链接](https://leetcode.cn/problems/transform-to-chessboard/)

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