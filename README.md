# Coding-Interviews


### 1、数组中重复的数字

题目链接： https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/

C++: 
```C++
//方法一：使用两层循环，改变了数组

class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        
        if(nums.size()<=0) return -1;
        for(int i=0;i<nums.size();i++){
            if(nums[i]<0 || nums[i]>nums.size()-1) 
               return -1;
        }

        for(int i=0;i<nums.size();i++){
            while(nums[i]!=i){
                if(nums[i]==nums[nums[i]]){
                    return nums[i];
                }
                swap(nums[i],nums[nums[i]]);
            }
        }

        return -1;
    }
};
```
Python:
```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        if len(nums)<=0:
            return -1

        for i in range(len(nums)):
            if nums[i]<0 or nums[i]>len(nums)-1:
                return -1
        
        for i in range(len(nums)):
            while nums[i]!=i:
                if nums[i]==nums[nums[i]]:
                    return nums[i]
                else:
                   temp = nums[i]
                   nums[i]=nums[temp]
                   nums[temp]=temp
        
        return -1
```


### 2、二维数组中的查找

题目链接：https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/

C++:
```C++
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
      if(matrix.size()<=0) return false;  
      int rows = matrix.size();
      int cols = matrix[0].size();
      if(matrix.size()>0 && matrix[0].size()>0){
          int row=0;
          int col=cols-1;
          while(row<rows && col>=0){
              if(matrix[row][col]==target){
                  return true;
              }
              else if(matrix[row][col]>target){
                  col--;
              }else{
                  row++;
              }
          }
      }
  
      return false;
    }
};
```
Python:
```Python
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        if len(matrix)<=0: return False
        row=0
        col=len(matrix[0])-1
        while row<len(matrix) and col >=0:
            if matrix[row][col]==target:
                return True
            elif matrix[row][col]>target:
                col-=1
            else:
                row+=1
        return False
```

### 3、替换空格

题目链接：https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/

C++:
```C++
class Solution {
public:
    string replaceSpace(string s) {
         int oldsize=s.size();
         int count=0;
         for(int i=0;i<oldsize;i++){
             if(s[i]==' ')
               count++;
         }

         s.resize(oldsize+count*2);
         int newsize=s.size();
         for(int i=newsize-1,j=oldsize-1;i>j;i--,j--){
             if(s[j]!=' '){
                 s[i]=s[j];
             }else{
                 s[i]='0';
                 s[i-1]='2';
                 s[i-2]='%';
                 i-=2;
             }
         }
         return s;
    }
};
```
Python:
```Python
class Solution:
    def replaceSpace(self, s: str) -> str:
        count=s.count(' ')
        oldsize=len(s)
        res=list(s)
        res.extend([' '] * count * 2)
        newsize=len(res)
        #双指针
        i=newsize-1
        j=oldsize-1
        while j>=0:
            if s[j]!=' ':
                res[i]=res[j]
            else:
                res[i]='0'
                res[i-1]='2'
                res[i-2]='%'
                i-=2
            j-=1
            i-=1
        return ''.join(res)
```

### 4、从尾到头打印链表

题目链接：https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/

C++:
```C++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */

 //方法一：使用栈
class Solution {
public:
    vector<int> reversePrint(ListNode* head) {
       vector<int> res;
       stack<ListNode*> st;
       ListNode* cur=head;
       while(cur!=NULL){
          st.push(cur);
          cur=cur->next;
       }
       while(!st.empty()){
         res.push_back(st.top()->val);
         st.pop();
       }
       return res;
    }
};

//方法二：回溯实现
class Solution{
public:
   void backtracking(ListNode* node,vector<int>& res){
       if(node==NULL) return;
       backtracking(node->next,res);
       res.push_back(node->val);
   }
   vector<int> reversePrint(ListNode* head) {
       vector<int> res;
       backtracking(head,res);
       return res;
   }
};
```
Python:
```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        res=[]
        while head:
            res.append(head.val)
            head=head.next
        return res[::-1]
```


### 5、重建二叉树

题目链接：https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/

C++:
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* traversal(vector<int>& preorder, vector<int>& inorder){
        if(preorder.size()==0) return NULL;
        int rootval=preorder[0];
        TreeNode* root=new TreeNode(rootval);
        if(preorder.size()==1) return root;

        int flag;
        for(flag=0;flag<inorder.size();flag++){
            if(inorder[flag]==rootval){
                break;
            }
        }

        vector<int> leftIn(inorder.begin(),inorder.begin()+flag);
        vector<int> rightIn(inorder.begin()+flag+1,inorder.end());

        vector<int> leftPre(preorder.begin()+1,preorder.begin()+1+leftIn.size());
        vector<int> rightPre(preorder.begin()+1+leftIn.size(),preorder.end());
        root->left=traversal(leftPre,leftIn);
        root->right=traversal(rightPre,rightIn);

        return root;
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
      if(preorder.size()==0 || inorder.size()==0) return NULL;
      return traversal(preorder,inorder);
    }
};
```
Python:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder: return None
        root = TreeNode(preorder[0])
        p=inorder.index(preorder[0])
        root.left=self.buildTree(preorder[1:p+1],inorder[:p])
        root.right=self.buildTree(preorder[p+1:],inorder[p+1:])
        return root
```


### 6、用两个栈实现队列

题目链接：https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/

C++:
```C++
class CQueue {
public:
    stack<int> stIn;
    stack<int> stOut;
    CQueue() {

    }
    
    void appendTail(int value) {
      stIn.push(value);
    }
    
    int deleteHead() {
      if(stOut.empty()&&stIn.empty()){
          return -1;
      }
      if(stOut.empty()){
          while(!stIn.empty()){
              stOut.push(stIn.top());
              stIn.pop();
          }
      }
      int res=stOut.top();
      stOut.pop();
      return res;
    }
};
```
Python:
```Python
class CQueue:

    def __init__(self):
        self.stIn=list()
        self.stOut=list()


    def appendTail(self, value: int) -> None:
        self.stIn.append(value)


    def deleteHead(self) -> int:
        if self.stOut==[] and self.stIn==[]:
            return -1
        if self.stOut==[]:
            while self.stIn:
                self.stOut.append(self.stIn[-1])
                self.stIn.pop()
        return self.stOut.pop()
```


### 7、斐波那契数列

题目链接：https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/

C++:
```C++
class Solution {
public:
    int fib(int n) {
      if(n<2) return n;
      int res=0;
      int a=0;
      int b=1;
      for(int i=2;i<=n;i++){
          res = (a+b)%1000000007;
          a=b;
          b=res;
      }
      return res;
    }
};
```
Python:
```Python
class Solution:
    def fib(self, n: int) -> int:
        if n<=1: return n
        a,b,c=0,1,0
        for i in range(1,n):
            c = (a+b)%1000000007
            a=b
            b=c
        return c
```


### 8、青蛙跳台阶问题

题目链接：https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/

C++:
```C++
class Solution {
public:
    int numWays(int n) {
       if(n==0) return 1;
       if(n<=2) return n;
       int res=0;
       int a=1;
       int b=1;
       for(int i=2;i<=n;i++){
           res = (a+b)%1000000007;
           a=b;
           b=res;
       }
       return res;

    }
};
```


### 9、旋转数组的最小数字

题目链接：https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/

C++:
```C++
class Solution {
public:
    int MinInorder(vector<int>& numbers, int left, int right){
        int res=numbers[left];
        for(int i=left+1;i<=right;i++){
            if(res>numbers[i]){
                res=numbers[i];
            }
        }
        return res;
    }
    int minArray(vector<int>& numbers) {
      if(numbers.size()<=0) return -1;
      int left=0;
      int right=numbers.size()-1;
      int mid = 0;
      while(numbers[left]>=numbers[right]){
        if(right-left==1){
            mid = right;
            break;
        }
        mid=left+((right-left)>>1);
        if(numbers[left]==numbers[right] && numbers[mid]==numbers[left]){
            return MinInorder(numbers,left,right);
        }
        if(numbers[mid]>=numbers[left]){
            left=mid;
        }else if(numbers[mid]<=numbers[right]){
            right=mid;
        }
      }
      return numbers[mid];
    }
};
```
Python:
```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        n=len(numbers)
        left,right,mid=0,n-1,0
        while numbers[left]>=numbers[right]:
            if right-left==1:
                return numbers[right]
            mid = (left+right)//2
            if numbers[left]==numbers[right] and numbers[left]==numbers[mid]:
                res=numbers[left]
                for i in range(left+1,right+1):
                    res=min(res,numbers[i])
                return res
            if numbers[mid]>=numbers[left]:
                left=mid
            elif numbers[mid]<=numbers[right]:
                right=mid
        return numbers[mid]
  ```
  
  
### 10、矩阵中的路径

题目链接：https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/

C++:
```C++
class Solution {
private:
    int rows,cols;
    bool backtracking(vector<vector<char>>& board, string word, int i, int j, int k){
        if(i>=rows || i<0 || j>=cols || j<0 || board[i][j]!=word[k]) return false;
        if(k==word.size()-1) return true;
        board[i][j]='\0';
        bool res = backtracking(board,word,i+1,j,k+1) || backtracking(board,word,i-1,j,k+1)||
                backtracking(board,word,i,j+1,k+1) || backtracking(board,word,i,j-1,k+1);
        board[i][j]=word[k];
        return res;
    }
public:
    bool exist(vector<vector<char>>& board, string word) {
       rows = board.size();
       cols = board[0].size();
       for(int i=0;i<rows;i++){
           for(int j=0;j<cols;j++){
               if(backtracking(board,word,i,j,0)) return true;
           }
       }
       return false;
    }
};
```
Python:
```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
 
        def backtracking(i,j,k):
            if i>=len(board) or i<0 or j>=len(board[0]) or j<0 or board[i][j]!=word[k]:
                return False
            if k==len(word)-1:
                return True
            
            board[i][j]=''
            res = backtracking(i+1,j,k+1) or backtracking(i-1,j,k+1) or backtracking(i,j-1,k+1) or backtracking(i,j+1,k+1)
            board[i][j]=word[k]

            return res

        for i in range(len(board)):
            for j in range(len(board[0])):
                if backtracking(i,j,0):
                    return True
        return False
 ```
 
 
 ### 11、机器人的运动范围
 
 题目链接：https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/
 
 C++:
 ```C++
 class Solution {
public:
    int getDightSum(int num){
        int sum=0;
        while(num>0){
            sum+=num%10;
            num/=10;
        }
        return sum;
    }
    int movingCount(int m, int n, int k) {
      if(m<=0 || n<=0 || k<0) return 0;
      int ans=1;
      vector<vector<int>> vis(m,vector<int>(n,0));
      vis[0][0]=1;
      for(int i=0;i<m;i++){
          for(int j=0;j<n;j++){
              if((i==0 && j==0)||(getDightSum(i)+getDightSum(j)>k)) continue;
              if(i-1>=0) vis[i][j] |=vis[i-1][j];
              if(j-1>=0) vis[i][j] |=vis[i][j-1];
              ans+=vis[i][j];
          }
      }
      return ans;
    }
};
```
Python:
```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        def getDigitSum(x):
            sum=0
            while x>0:
                sum+=x%10
                x//=10
            return sum
        
        if m<=0 or n<=0 or k<0:
            return 0
        vis=[[ 0 for i in range(n)] for j in range(m)]
        vis[0][0]=1
        ans=1
        for i in range(m):
            for j in range(n):
                if (i==0 and j==0) or (getDigitSum(i)+getDigitSum(j)>k):
                    continue
                if i-1>=0:
                    vis[i][j] |= vis[i-1][j]
                if j-1>=0:
                    vis[i][j] |= vis[i][j-1]
                ans+=vis[i][j]
        return ans
```


### 12、剪绳子

题目链接：https://leetcode-cn.com/problems/jian-sheng-zi-lcof/

C++:
```C++
//动态规划
class Solution {
public:
    int cuttingRope(int n) {
      if(n<2) return 0;
      if(n==2) return 1;
      if(n==3) return 2;
      vector<int> dp(n+1);
      dp[0]=0;
      dp[1]=1;
      dp[2]=2;
      dp[3]=3;
      int res=0;
      for(int i=4;i<=n;i++){
          res=0;
          for(int j=1;j<=i/2;j++){
              res=max(res,dp[j]*dp[i-j]);
              dp[i]=res;
          }
      }
      return dp[n];
    }
};


//贪婪算法
class Solution{
public:
    int cuttingRope(int n){
        if(n<2) return 0;
        if(n==2) return 1;
        if(n==3) return 2;

        int times3 = n/3;
        if(n-times3*3==1){
            times3-=1;
        }

        int times2 = (n-times3*3)/2;
        return (int)(pow(3,times3))*(int)(pow(2,times2));
    }
};
```
Python:
```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        if n<2: return 0
        if n==2: return 1
        if n==3: return 2

        dp=[0]*(n+1)
        dp[0]=0
        dp[1]=1
        dp[2]=2
        dp[3]=3

        for i in range(4,n+1):
            res=0
            for j in range(1,i//2+1):
                res=max(res,dp[j]*dp[i-j])
                dp[i]=res
        
        return dp[n]
```


### 13、剪绳子II

题目链接：https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/submissions/

C++:
```C++
class Solution {
public:
    int cuttingRope(int n) {
        if(n < 4){
            return n - 1;
        }
        long ans = 1, t = n % 3;
        int mod = 1000000007;
        for(int i = 1; i < n / 3;i++){
            ans = 3 * ans % mod;
        }
        return (ans * 3 * 4 / (4 - t)) % mod;
    }
};
```


### 14、二进制中1的个数

题目链接：https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/

C++:
```C++
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int count = 0;
        while(n){
            count++;
            n=(n-1)&n;
        }
        return count;
    }
};
```
Python:
```Python
class Solution:
    def hammingWeight(self, n: int) -> int:
        res=0
        while n:
            res+=1
            n=(n-1)&n
        return res
```


### 15、数值的整数次方

题目链接：https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/

C++:
```C++
class Solution {
public:
    double myPow(double x, int n) {
      if(x==0.0) return 0.0;
      if(n==-1) return 1/x;
      if(n==0) return 1;
      if(n==1) return x;

      double res=myPow(x,n>>1);
      res*=res;
      if(n&0x1==1){
         res *= x;
      } 
      return res;
    }
};
```
Python:
```Python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x==0.0: return 0.0
        if n==-1: return 1/x
        if n==0: return 1
        if n==1: return x
        res =self.myPow(x,n>>1)
        res *= res
        if n&0x1==1:
            res *= x
        return res
```


### 16、打印从1到最大的n位数

题目链接：https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/

C++:
```C++

class Solution{
private:
   vector<string> res;
   string s;
   char num[10]={'0','1','2','3','4','5','6','7','8','9'};

   //生成长度为len的数，固定第x位
   void dfs(int x, int len){
       if(x==len){
           res.push_back(s);
           return;
       }
       //如果固定是第0位，表明是第1位，不能取0
       int start = x==0 ? 1: 0;
       for(int i=start;i<10;i++){
           s.push_back(num[i]);
           dfs(x+1,len);
           s.pop_back();
       }
   }
public:
   vector<int> printNumbers(int n){
       for(int i=1;i<=n;i++){
           dfs(0,i);
       }
       
       vector<int> ans;
       for(int i=0;i<res.size();i++){
           ans.push_back(stoi(res[i]));
       }
       return ans;
   }

};
```


### 17、删除链表的节点

题目链接：https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/

C++:
```C++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        if(head->val==val) return head->next;
        ListNode *pre = head;
        ListNode *cur = head->next;
        while(cur!=nullptr && cur->val!=val){
            pre=cur;
            cur=cur->next;
        }
        if(cur!=nullptr){
            pre->next=cur->next;
        }
        return head;

    }
};
```
Python:
```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if head.val==val: return head.next
        pre=head
        cur=head.next
        while cur!=None and cur.val!=val:
            pre=cur
            cur=cur.next
        if cur!=None:
            pre.next=cur.next
        return head
```


### 18、表示数值的字符串

题目链接：https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/

C++:
```C++
class Solution {
private:
    int index = 0;
    bool scanUnsignedInteger(string s){
        int start = index;
        while(index<s.size() && s[index]>='0' && s[index]<='9'){
            index++;
        }
        return start<index;
    }

    bool scanInteger(string s){
        if(index<s.size() && (s[index]=='+' || s[index]=='-')){
            index++;
        }
        return scanUnsignedInteger(s);
    }

public:
    bool isNumber(string s) {
      // 首先去掉首尾空格
      s.erase(0,s.find_first_not_of(" "));  
      s.erase(s.find_last_not_of(" ") + 1);
      

      if(s.size()<1) return false;

      bool numeric = scanInteger(s);

      if(index<s.size() && s[index]=='.'){
          index++;
          numeric = scanUnsignedInteger(s) || numeric;
      }

      if(index<s.size() && (s[index]=='e' || s[index]=='E')){
          index++;
          numeric = numeric && scanInteger(s);
      }

      return numeric && index==s.size();
    }
};
```
Python:
```python
class Solution:
    def isNumber(self, s: str) -> bool:
       
        s=s.strip() #去除首尾空格
        if len(s)<1:return False

        #判断整数的正负号
        def scanInteger(s):
            nonlocal index
            if index<len(s) and (s[index]=='+' or s[index]=='-'):
                index+=1
            return scanUnsignedInteger(s)
        
        #判断整数数值部分
        def scanUnsignedInteger(s):
            nonlocal index
            start = index
            while index<len(s) and s[index]>='0' and s[index]<='9':
                index+=1
            return start<index

        index = 0
        numeric = scanInteger(s)
        if index<len(s) and s[index]=='.':
            index+=1
            numeric = scanUnsignedInteger(s) or numeric
        if index<len(s) and (s[index]=='e' or s[index]=='E'):
            index+=1
            numeric = scanInteger(s) and numeric
        
        return numeric and (index==len(s))
```


### 19、调整数组顺序使得奇数位于偶数的前面

题目链接：https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/

C++:
```C++
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {

        if(nums.size()<=0) return vector<int>(0);

        int left=0;
        int right=nums.size()-1;
        while(left<right){
            //向后移动left,直到它指向偶数
            while(left<right && (nums[left]&0x1)!=0){
                left++;
            }

            //向前移动right,直到它指向奇数
            while(left<right && (nums[right]&0x1)==0){
                right--;
            }
            swap(nums[left],nums[right]);
        }
        return nums;

    }
};
```
Python:
```Python
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        if len(nums)<=0: return []
        left=0
        right=len(nums)-1
        while left<right:
            while left<right and nums[left]&0x1!=0:
                left+=1
            while left<right and nums[right]&0x1==0:
                right-=1
            nums[left],nums[right]=nums[right],nums[left]
        return nums
```


### 20、链表中倒数第k个节点

题目链接：https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/

C++:
```C++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        if(head==NULL||k==0){
            return NULL;
        }
        ListNode* pre = head;
        ListNode* cur = head;

        for(int i=0;i<k;i++){
            if(cur==NULL){
                return NULL;
            }
            cur=cur->next;
        }
        while(cur!=NULL){
            pre=pre->next;
            cur=cur->next;
        }
        return pre;

    }
};
```
Python:
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        if head==None or k==0: return None
        fast,slow=head,head
        for i in range(k):
            if fast:
                fast=fast.next
            else:
                return None
        
        while fast:
            fast=fast.next
            slow=slow.next
        
        return slow
 ```
 
 
 ### 21、反转链表
 
 题目链接：https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/
 
 C++:
 ```C++
 /**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre=NULL;
        ListNode* tmp;
        ListNode* cur=head;
        while(cur){
            tmp=cur->next;
            cur->next=pre;
            pre=cur;
            cur=tmp;
        }
        return pre;
        }
};
```
Python:
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre=None
        cur=head
        while cur:
            tmp=cur.next
            cur.next=pre
            pre=cur
            cur=tmp
        return pre
```


### 22、合并两个排序的链表

题目链接：https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/

C++:
```C++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1==NULL){
            return l2;
        }else if(l2==NULL){
            return l1;
        }

        ListNode* cur=NULL;
        if(l1->val<l2->val){
            cur=l1;
            cur->next=mergeTwoLists(l1->next,l2);
        }else{
            cur=l2;
            cur->next=mergeTwoLists(l1,l2->next);
        }
        return cur;

    }
};
```
Python:
```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1==None:
            return l2
        elif l2==None:
            return l1
        
        cur=None
        if l1.val<l2.val:
            cur=l1
            cur.next=self.mergeTwoLists(l1.next,l2)
        else:
            cur=l2
            cur.next=self.mergeTwoLists(l1,l2.next)
        
        return cur
```


### 23、树的子结构

题目链接：https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/

C++:
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isRec(TreeNode* A, TreeNode* B) {
       if(B==NULL) return true;
       if(A==NULL) return false;
       if(A->val!=B->val) return false;
       return isRec(A->left,B->left)&&isRec(A->right,B->right);
    }
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        return A!=NULL && B!=NULL && (isRec(A,B) || isSubStructure(A->left,B) || isSubStructure(A->right,B));
    }
};
```
Python：
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        def isRec(A,B):
            if B==None:return True
            if A==None:return False
            if A.val!=B.val: return False
            return isRec(A.left,B.left) and isRec(A.right,B.right)
        
        return bool(A and B) and(isRec(A,B) or self.isSubStructure(A.left,B) or self.isSubStructure(A.right,B))
```


### 23、二叉树的镜像

题目链接：https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/

C++:
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
      if(root==NULL) return NULL;
      if(root->left==NULL && root->right==NULL) return root;

      swap(root->left,root->right); //根
      if(root->left){
          mirrorTree(root->left);   //左
      }
      if(root->right){
          mirrorTree(root->right);  //右
      }
      return root;
    }
};
```
Python:
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if root==None:return None
        if root.left==None and root.right==None: return root

        root.left,root.right=root.right,root.left
        if root.left:
            self.mirrorTree(root.left)
        if root.right:
            self.mirrorTree(root.right)
        
        return root
```


### 24、对称的二叉树

题目链接：https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/

C++:
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(root==NULL)  return true;
        return Compare(root->left,root->right);
    }

    bool Compare(TreeNode* curleft, TreeNode* curright){
        if(curleft==NULL && curright==NULL){
            return true;
        }
        if(curleft==NULL && curright!=NULL){
            return false;
        }
        if(curleft!=NULL&& curright==NULL){
            return false;
        }

        if(curright->val!=curleft->val) return false;
        return Compare(curleft->left,curright->right) && Compare(curleft->right,curright->left);

    }
};
```
Python:
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if root==None: return True
        def Compare(a,b):
            if a==None and b==None: return True
            if a!=None and b==None: return False
            if a==None and b!=None: return False
            if a.val!=b.val: return False
            return Compare(a.left, b.right) and Compare(a.right,b.left)
        return Compare(root.left,root.right)
```


### 25、顺时针打印矩阵

题目链接：https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/

C++:
```C++
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> res;
        if(matrix.size()<=0 || matrix[0].size()<=0) return res;
        int rows = matrix.size();
        int cols = matrix[0].size();

        int left=0,right=cols-1;
        int up=0,down=rows-1;
    
        while(left<=right && up<=down){
             
             for(int i=left;i<right+1;i++){
                 res.push_back(matrix[up][i]);
             }
             for(int i=up+1;i<down+1;i++){
                 res.push_back(matrix[i][right]);
             }

             if(left<right && up<down){
                 for(int i=right-1;i>left;i--){
                     res.push_back(matrix[down][i]);
                 }
                 for(int i=down;i>up;i--){
                     res.push_back(matrix[i][left]);
                 }

             }

             left++;
             right--;
             up++;
             down--;
        }
        return res;


    }
};
```


### 26、包含min函数的栈

题目链接：https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/

C++:
```C++
class MinStack {
public:
    /** initialize your data structure here. */
    stack<int> st1;
    stack<int> st2;
    MinStack() {
      
    }
    
    void push(int x) {
      st1.push(x);
      if(st2.size()==0 || x<st2.top()){
          st2.push(x);
      }else{
          st2.push(st2.top());
      }
    }
    
    void pop() {
        if(st1.size()>0 && st2.size()>0){
           st1.pop();
           st2.pop(); 
        }else{
            return;
        }
      
    }
    
    int top() {
        return st1.top();
    }
    
    int min() {
        return st2.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->min();
 */
 ```


### 27、栈的压入，弹出序列

题目链接：https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/

C++:
```C++
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        if(pushed.size()==0 && popped.size()==0) return true;
        if(pushed.size() != 0 && popped.size() != 0){
            stack<int> data;
            int j=0;
            for(int i=0;i<pushed.size();i++){
                data.push(pushed[i]);
                while(!data.empty() && data.top()==popped[j]){
                    data.pop();
                    j++;
                }
            }
            return data.empty();
        }
        return false;
    }
};
```
Python:
```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        st=[]
        i=0
        for num in pushed:
            st.append(num)
            while st and st[-1]==popped[i]:
                st.pop()
                i+=1
        return not st
```


### 28、从上到下打印二叉树II

题目链接：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/

C++:
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> que;
        if(root!=NULL) que.push(root);
        vector<vector<int>> res;
        while(!que.empty()){
            int size=que.size();
            vector<int> vec;
            for(int i=0;i<size;i++){
                TreeNode* node = que.front();
                vec.push_back(node->val);
                que.pop();
                if(node->left){
                    que.push(node->left);
                }
                if(node->right){
                    que.push(node->right);
                }
            }
            res.push_back(vec);
        }
        return res;
    }
};
```
Python:
```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        
        que=[root]
        res=[]

        while que:
            size=len(que)
            vec=[]
            for i in range(size):
                node = que.pop(0)
                vec.append(node.val)
                if node.left:
                    que.append(node.left)
                if node.right:
                    que.append(node.right)
            res.append(vec)
        return res
```


### 29、从上到下打印二叉树III

题目链接：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/

C++:
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        //使用两个栈
        stack<TreeNode*> st1;
        stack<TreeNode*> st2;
        vector<vector<int>> res;

        if(root!=NULL) st1.push(root);
        int count = 0;
        while(!st1.empty() || !st2.empty()){
            vector<int> vec;
            //偶数层时，从右到左打印，则入栈的顺序为：根->左节点->右节点
            if(count%2==0){
                int len = st1.size();
                for(int i=0;i<len;i++){
                    TreeNode* cur1 = st1.top();
                    vec.push_back(cur1->val);
                    st1.pop();
                    if(cur1->left) st2.push(cur1->left);
                    if(cur1->right) st2.push(cur1->right);
                }
            }else{
                //奇数层时，从左到右打印，则入栈的顺序为: 根->右节点->左节点
                int len2 = st2.size();
                for(int i=0;i<len2;i++){
                    TreeNode* cur2 = st2.top();
                    vec.push_back(cur2->val);
                    st2.pop();
                    if(cur2->right) st1.push(cur2->right);
                    if(cur2->left) st1.push(cur2->left);
                }
            }
            count++;
            res.push_back(vec);
        }
        return res;
    }
};
```
Python：
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        res=[]
        st1=[root]
        st2=[]
        count = 0
        while (st1!=[]) or (st2!=[]):
            vec=[]
            if count%2==0:
                len1 = len(st1)
                for i in range(len1):
                    node1 = st1.pop(-1)
                    vec.append(node1.val)
                    if node1.left: st2.append(node1.left)
                    if node1.right: st2.append(node1.right)
            else:
                len2 = len(st2)
                for i in range(len2):
                    node2 = st2.pop(-1)
                    vec.append(node2.val)
                    if node2.right: st1.append(node2.right)
                    if node2.left: st1.append(node2.left)
            
            count+=1
            res.append(vec)

        return res
```


### 30、二叉搜索树的后序遍历序列

题目链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/

C++:
```C++
class Solution {
public:
    bool verifyPostorder(vector<int>& postorder) {
        int len = postorder.size();
        if(len<=0) return true;
        int root = postorder[len-1];
        //在二叉搜索树中左子树节点的值小于根节点的值
        int i=0;
        for(;i<len-1;i++){
            if(postorder[i]>root){
                break;
            }
        }
        //在二叉搜索树中右子树的值大于根节点的值
        int j = i;
        for(;j<len-1;j++){
            if(postorder[j]<root){
                return false;
            }
        }
        bool left = true;
        bool right = true;
        vector<int> vl(postorder.begin(),postorder.begin()+i);
        vector<int> vr(postorder.begin()+i,postorder.end()-1);
        //判断左子树是不是二叉搜索树
        if(i>0) left = verifyPostorder(vl);
        //判断右子树是不是二叉搜索树
        if(i<len-1) right = verifyPostorder(vr);
        return left&&right;

    }
};
```
Python:
```python
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        n = len(postorder)
        if n<=0: return True
        root = postorder[n-1]
        i=0
        while i<n-1:
            if postorder[i]>root:
                break
            i+=1
        
        j=i
        while j<n-1:
            if postorder[j]<root:
                return False
            j+=1
        
        left = True
        right = True
        if i>0:
            left = self.verifyPostorder(postorder[0:i])
        if i<n-1:
            right = self.verifyPostorder(postorder[i:n-1])
        
        return left and right
```

### 31、二叉树中和为某一值的路径

题目链接：https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/

C++:
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;
    void backtracking(TreeNode* cur, int count){
        if(!cur->left && !cur->right && count==0){
            res.push_back(path);
            return;
        }

        if(!cur->left && !cur->right){
            return;
        }

        if(cur->left){
            count-=cur->left->val;
            path.push_back(cur->left->val);
            backtracking(cur->left,count);
            count+=cur->left->val;
            path.pop_back();
        }

        if(cur->right){
            count-=cur->right->val;
            path.push_back(cur->right->val);
            backtracking(cur->right,count);
            count+=cur->right->val;
            path.pop_back();
        }
        return;

    }
public:
    vector<vector<int>> pathSum(TreeNode* root, int target) {
        if(root==NULL) return res;
        path.push_back(root->val);
        backtracking(root,target-root->val);
        return res;

    }
};
```
Python:
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        res=[]
        path=[]
        
       
        def backtracking(cur,count):
            if (not cur.left) and (not cur.right) and count==0:
                res.append(path[:])
                return 
            
            if (not cur.left) and (not cur.right):
                return 

            if cur.left:
                count-=cur.left.val
                path.append(cur.left.val)
                backtracking(cur.left,count)
                count+=cur.left.val
                path.pop()
            
            if cur.right:
                count-=cur.right.val
                path.append(cur.right.val)
                backtracking(cur.right,count)
                count+=cur.right.val
                path.pop()
            
            return 

        if root==None: return res
        path.append(root.val)
        backtracking(root,target-root.val)
        return res
```

### 32、复杂链表的复制

题目链接：https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/

C++:
```C++
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
*/
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(head==NULL) return NULL;

        Node* cur = head;
        //第一步：遍历链表，复制每个节点，将复制节点放在原节点的后面
       for(Node* node = head;node!=NULL;node = node->next->next){
           Node* nodeNew = new Node(node->val);
           nodeNew->next = node->next;
           node->next = nodeNew;
       }

        //第二步：重新遍历链表，复制旧节点的随机指针给新节点
        for(Node* node = head; node!=NULL; node = node->next->next){
            Node* nodeNew = node->next;
            nodeNew->random = (node->random!=NULL) ? node->random->next : NULL;
        }

        //第三步：拆分链表，将链表分成原链表和复制后的链表
        Node* headNew = head->next;
        for(Node* node = head; node!=NULL; node = node->next){
            Node* nodeNew = node->next;
            node->next = node->next->next;
            nodeNew->next = (nodeNew->next!=NULL) ? nodeNew->next->next : NULL;
        }
        return headNew;
    }
};
```


### 33、二叉搜索树与双向链表

题目链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/

C++:
```C++
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;

    Node() {}

    Node(int _val) {
        val = _val;
        left = NULL;
        right = NULL;
    }

    Node(int _val, Node* _left, Node* _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
private:
    Node* pre;
    Node* head;
    void dfs(Node* cur){
        if(cur==nullptr) return;
        dfs(cur->left);
        if(pre!=nullptr) pre->right = cur;
        else head = cur;
        cur->left = pre;
        pre = cur;
        dfs(cur->right);
    }
public:
    Node* treeToDoublyList(Node* root) {
        if(root==nullptr) return nullptr;
        dfs(root);
        head->left = pre;
        pre->right = head;
        return head;
        
    }
};
```
Python:
```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':

        def dfs(cur):
            if not cur: return
            dfs(cur.left)
            if self.pre: self.pre.right = cur
            else: self.head = cur
            cur.left = self.pre
            self.pre = cur
            dfs(cur.right)

        if not root: return
        self.pre = None
        self.head = None
        dfs(root)
        self.head.left = self.pre
        self.pre.right = self.head
        return self.head
```


### 34、序列化二叉树

题目链接：https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/

C++:
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Codec {
public:
    void rserialize(TreeNode* root, string& str) {
        if (root == nullptr) {
            str += "None,";
        } else {
            str += to_string(root->val) + ",";
            rserialize(root->left, str);
            rserialize(root->right, str);
        }
    }

    string serialize(TreeNode* root) {
        string ret;
        rserialize(root, ret);
        return ret;
    }

    TreeNode* rdeserialize(list<string>& dataArray){
        if(dataArray.front()=="None"){
            dataArray.erase(dataArray.begin());
            return nullptr;
        }

        TreeNode* root = new TreeNode(stoi(dataArray.front()));
        dataArray.erase(dataArray.begin());
        root->left = rdeserialize(dataArray);
        root->right = rdeserialize(dataArray);
        return root;
    }
    TreeNode* deserialize(string data) {
        list<string> dataArray;
        string str;
        for(auto& ch : data){
            if(ch==','){
                dataArray.push_back(str);
                str.clear();
            }else{
                str.push_back(ch);
            }
        }
        if(!str.empty()){
            dataArray.push_back(str);
            str.clear();
        }
        return rdeserialize(dataArray);
        
    }
};

// Your Codec object will be instantiated and called as such:
// Codec codec;
// codec.deserialize(codec.serialize(root));
```


### 35、字符串的排列

题目链接：https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/

C++:
```C++
class Solution {
private:
    vector<string> res;
    string path;
    void backtracking(const string& s,vector<bool> used){
        if(path.size()==s.size()){
            res.push_back(path);
            return;
        }

        for(int i=0;i<s.size();i++){
            if(i>0 && used[i-1]==false && s[i]==s[i-1]){
                continue;
            }
            if(used[i]==false){
                used[i]=true;
                path.push_back(s[i]);
                backtracking(s,used);
                used[i]=false;
                path.pop_back();

            }
        }
    }
public:
    vector<string> permutation(string s) {
        vector<bool> used(s.size(),false);
        sort(s.begin(),s.end());
        backtracking(s,used);
        return res;

    }
};
```


### 36、数组中出现次数超过一半的数字

题目链接：https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/

C++:
```C++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int cond = -1;
        int cnt = 0;
        for(int i=0;i<nums.size();i++){
            if(cnt==0){
                cond = nums[i];
                cnt++;
            }else{
                if(cond==nums[i]){
                    cnt++;
                }else{
                    cnt--;
                }
            }
        }
        return cond;
    }
};
```
Python:
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        cond = -1
        cnt = 0
        for i in nums:
            if cnt==0:
                cond = i
                cnt += 1
            else:
                if cond==i:
                    cnt+=1
                else:
                    cnt-=1
        return cond
```

### 37、最小的K个数

题目链接：https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/

C++:
```C++
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int> vec(k,0);
        if(k==0) return vec;

        priority_queue<int> Q;
        for(int i=0;i<k;i++){
            Q.push(arr[i]);
        }

        for(int i=k;i<arr.size();i++){
            if(Q.top()>arr[i]){
                Q.pop();
                Q.push(arr[i]);
            }
        }

        for(int i=0;i<k;i++){
            vec[i] = Q.top();
            Q.pop();
        }
        return vec;

    }
};
```
Python：
```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        if k==0: return []

        hp = [-x for x in arr[:k]]
        heapq.heapify(hp)
        for i in range(k, len(arr)):
            if -hp[0]>arr[i]:
                heapq.heappop(hp)
                heapq.heappush(hp,-arr[i])
        ans = [-x for x in hp]
        return ans
```


### 38、数据流中的中位数

题目链接：https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/

C++:
```C++
class MedianFinder {
public:
    /** initialize your data structure here. */
    //创建大顶堆和小顶堆
    priority_queue<int, vector<int>, less<int>> maxHeap;
    priority_queue<int, vector<int>, greater<int>> minHeap;

    MedianFinder() {

    }
    
    void addNum(int num) {
        if(maxHeap.size()==minHeap.size()){
            maxHeap.push(num);
            minHeap.push(maxHeap.top());
            maxHeap.pop();
        }else{
            minHeap.push(num);
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }

    }
    
    double findMedian() {
        if(maxHeap.size()==minHeap.size()){
            return (minHeap.top()+maxHeap.top())/2.0;
        }else{
            return minHeap.top();
        }

    }
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */
 ```
 
 ### 39、连续子数组的最大和
 
 题目链接：https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/
 
 C++:
 ```C++
 class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int res = INT32_MIN;
        int count = 0;
        for(int i=0;i<nums.size();i++){
            count += nums[i];
            res = max(res,count);
            if(count<=0) count = 0;
        }

    }
};
```

### 40、1~n整数中1出现的次数

题目链接：https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/

C++:
```C++
class Solution {
public:
    int countDigitOne(int n) {
        long long mulk = 1;
        int ans = 0;
        for (int k = 0; n >= mulk; ++k) {
            ans += (n / (mulk * 10)) * mulk + min(max(n % (mulk * 10) - mulk + 1, 0LL), mulk);
            mulk *= 10;
        }
        return ans;
    }
};
```


### 44、数字序列中某一位的数字

题目链接：https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/

C++:
```C++
class Solution {
public:
    int findNthDigit(int n) {
       if(n<10){
           return n;
       }

       int digit = 1;
       while(true){
           //计算出得到digit位的数字，总共有多少个
           long long num = countInteger(digit);
           //判断n是否位于某digit位数之中
           if(n<num*digit){
               //找到那个数字
               return digitIndex(n,digit);
           }
           n -= digit * num;
           digit++;
       }
       return -1;
    }
    //计算digit位需要多少数字，输入2，返回（10-99）的个数90
    int countInteger(int digit){
        if(digit==1) return 10;
        int count =(int)pow(10,digit-1);
        return 9*count;
    }
    //找到数字
    int digitIndex(int n, int digit){
        //digit位的第一个数 + n/digit
        int num = beginNumber(digit)+n/digit;
        int temp = digit - n%digit;
        for(int i=1;i<temp;i++){
            num /= 10;
        }
        return num%10;
    }
    //找到digit位的第一个数字，输入2，返回10，输入3，返回100
    int beginNumber(int digit){
        if(digit==1){
            return 0;
        }

        return (int)pow(10,digit-1);
    }
};
```
Python:
```
class Solution:
    def findNthDigit(self, n: int) -> int:
        if n<10:
            return n

        def countInteger(digit):
            if digit==1: return 10
            return 9*math.pow(10,digit-1)
        
        def digitIndex(n,digit):
            if digit==1:
                num = n//digit
            else:
                num = math.pow(10,digit-1) + n//digit
            temp = digit - n%digit
            for i in range(1,int(temp)):
                num /= 10
            return int(num%10)

        digit = 1
        while True:
            num = countInteger(digit)
            if n<num*digit:
                return digitIndex(n,digit)
            n -= digit*num
            digit+=1
        
        return -1
```


### 45、把数组排成最小的数

题目链接：https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/

C++:
```C++
class Solution {
public:
    void quickSort(vector<string>& strs, int left ,int right){
        if(left>=right) return;
        int i=left,j=right;
        while(i<j){
            //定义排序规则
            //若拼接字符串 x + y > y + x，则x大于y，反之小于
            
            while(strs[j]+strs[left]>=strs[left]+strs[j] && i<j) j--;
            while(strs[i]+strs[left]<=strs[left]+strs[i] && i<j) i++;
            swap(strs[i],strs[j]);
        }

        swap(strs[i],strs[left]);
        quickSort(strs,left,i-1);
        quickSort(strs,i+1,right);
    }
    string minNumber(vector<int>& nums) {
        vector<string> strs;
        for(int i=0;i<nums.size();i++){
           strs.push_back(to_string(nums[i]));
        }
        //使用快排
        quickSort(strs,0,strs.size()-1);
        string res;
        for(string s: strs){
            res+=s;
        }
        return res;
    }
};
```

### 46、把数字翻译成字符串

题目链接：https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/

C++:
```c++
class Solution {
public:
    int translateNum(int num) {
        string s = to_string(num);
        int n = s.size();
        vector<int> dp(n+1, 0);
        dp[0] = 1;
        for(int i=1;i<=n;i++){
            dp[i] = dp[i-1];
            if(i>1){
                int t = (s[i-2]-'0')*10+s[i-1]-'0';
                if(t>=10 && t<=25){
                    dp[i] += dp[i-2];
                }
            }
        }
        return dp[n];

    }
};
```
Python：
```python
class Solution:
    def translateNum(self, num: int) -> int:
        s = str(num)
        n = len(s)
        dp = [0 for i in range(n+1)]
        dp[0] = 1

        for i in range(1,n+1):
            dp[i] = dp[i-1]
            if i>1:
                t = int(s[i-2])*10+int(s[i-1])
                if t>=10 and t<=25:
                    dp[i] += dp[i-2]
        
        return dp[n]
```


### 47、礼物的最大价值

题目链接：https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/

C++:
```C++
class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
      if(grid.size()<=0 || grid[0].size()<=0){
          return 0;
      }

    for(int i=0;i<grid.size();i++){
        for(int j=0;j<grid[0].size();j++){
            if(i==0 && j==0) continue;
            if(i==0 && j!=0) grid[i][j]+=grid[i][j-1];
            else if(j==0) grid[i][j]+=grid[i-1][j];
            else grid[i][j] += max(grid[i-1][j],grid[i][j-1]);
        }
    }
    return grid[grid.size()-1][grid[0].size()-1];
    }
};
```
Python：
```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        for i in range(rows):
            for j in range(cols):
                if i==0 and j==0: continue
                if i==0 and j!=0: grid[i][j] += grid[i][j-1]
                if i!=0 and j==0: grid[i][j] += grid[i-1][j]
                if i!=0 and j!=0: grid[i][j] += max(grid[i-1][j],grid[i][j-1])

        return grid[-1][-1]
```


### 48、最长不含重复字符的子字符串

题目链接：https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/

C++:
```C++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.empty()) return 0;
        if(s.size()==1) return 1;
        //使用滑动窗口法
        int res=0;
        int left=-1;
        //哈希表记录的是前面字符出现的位置
        unordered_map<char,int> map;
        //遍历字符数组
        for(int i=0;i<s.size();i++){
            //如果出现和前面重复的字符时
            if(map.find(s[i])!=map.end()){
                //left指针移到此字符出现的位置
                left = max(map[s[i]],left);
            }
            //记录字符出现的位置
            map[s[i]]=i;
            //如果没有出现重复的字符，这不含重复字符串的长度为i-left
            res=max(res,i-left);
        }
        return res;
    }
};
```
Python:
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        hashmap = {}
        left = -1
        res = 0
        for i in range(len(s)):
            if s[i] in hashmap:
                left = max(hashmap[s[i]],left)
            hashmap[s[i]]=i
            res = max(res,i-left)
        return res
```


### 49、丑数

题目链接：https://leetcode-cn.com/problems/chou-shu-lcof/

C++:
```C++
class Solution {
public:
    int nthUglyNumber(int n) {
        if(n<=6) return n;
        vector<int> res(n,0);
        res[0]=1;
        int p2=0,p3=0,p5=0;
        for(int i=1;i<n;i++){
            res[i] = min(res[p2]*2,min(res[p3]*3,res[p5]*5));
            if(res[i]==res[p2]*2) p2++;
            if(res[i]==res[p3]*3) p3++;
            if(res[i]==res[p5]*5) p5++;
        }
        return res[n-1];
    }
};
```
Python:
```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        if n<=6: return n
        res=[0 for i in range(n)]
        res[0]=1
        p2,p3,p5=0,0,0
        for i in range(1,n):
            res[i] = min(res[p2]*2,res[p3]*3,res[p5]*5)
            if res[i]==res[p2]*2: p2+=1
            if res[i]==res[p3]*3: p3+=1
            if res[i]==res[p5]*5: p5+=1
        return res[n-1]
```


### 50、第一个只出现一次的字符

题目链接：https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/

C++:
```C++
class Solution {
public:
    char firstUniqChar(string s) {
      if(s.size()<=0) return ' ';
      vector<int> fre(256,0);
      for(int i=0;i<s.size();i++){
          fre[s[i]]++;
      }
      for(int j=0;j<s.size();j++){
          if(fre[s[j]]==1){
              return s[j];
          }
      }
      return ' ';
    }
};
```
Python：
```python
class Solution:
    def firstUniqChar(self, s: str) -> str:
        fre = collections.Counter(s)
        for i,ch in enumerate(s):
            if fre[ch] == 1:
                return ch
        return ' '
```


### 51、数组中的逆序对

题目链接：https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/


C++:
```C++
class Solution {
public:
    int reversePairs(vector<int>& nums) {
        if(nums.size()<=0) return 0;
        //使用归并排序法统计逆序对
        return MergeSort(nums,0,nums.size()-1);
    }

    int MergeSort(vector<int>& nums,int start, int end){
        if(start==end) return 0;
        
        int mid = (start+end)/2;
        //1、对左半部分递归，得到左边的逆序对数
        int left = MergeSort(nums,start,mid);
        //2、对右半部分递归, 得到右边的逆序对数
        int right = MergeSort(nums,mid+1,end);
        int i = mid,j=end;
        //3、统计混排得到的逆序对的个数
        
        //copy数组用于存储混排得到的逆序对
        vector<int> copy(end-start+1);
    
        int indexCopy = end-start;
        int count = 0;
        //遍历数组
        while(i>=start && j>=mid+1){
            //如果前面的数大于后面的数
            if(nums[i]>nums[j]){
                //将前面的数组向前移
                copy[indexCopy--]=nums[i--];
                //逆序对数 = 后面数组中剩余数字的个数
                count += j-mid;
            }else{
                //将后面的数组向前移，此时没有产生逆序对
                copy[indexCopy--]=nums[j--];
            }
        }
        //下面是对数组进行排序
        while(i>=start){
            copy[indexCopy--] = nums[i--];
        }
        while(j>=mid+1){
            copy[indexCopy--] = nums[j--];
        }

        i = 0;
        while(start<=end){
            nums[start++] = copy[i++];
        }
        //总逆序对数 = 1+2+3
        return left+right+count;

    }
};
```
Python:
```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        if len(nums)<=0: return 0

        def MergeSort(nums,start,end):
            if start==end: 
                return 0
            mid = (start+end)//2
            left = MergeSort(nums,start,mid)
            right = MergeSort(nums,mid+1,end)
            i,j=mid,end
            copy=[0 for i in range(end-start+1)]
            copyIndex = end-start
            count = 0
            while i>=start and j>=mid+1:
                if nums[i]>nums[j]:
                    copy[copyIndex]=nums[i]
                    copyIndex-=1
                    i-=1
                    count += j-mid
                else:
                    copy[copyIndex]=nums[j]
                    copyIndex-=1
                    j-=1
            
            while i>=start:
                copy[copyIndex]=nums[i]
                copyIndex-=1
                i-=1
            
            while j>=mid+1:
                copy[copyIndex]=nums[j]
                copyIndex-=1
                j-=1
            i=0
            while start<=end:
                nums[start]=copy[i]
                i+=1
                start+=1
            
            return left+right+count

        return MergeSort(nums,0,len(nums)-1)
```

### 52、两个链表的第一个公共节点

题目链接：https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/

C++:
```C++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode* curA = headA;
        ListNode* curB = headB;
        int lenA = 0,lenB = 0;
        while(curA!=NULL){
            lenA++;
            curA = curA->next;
        }
        while(curB!=NULL){
            lenB++;
            curB = curB->next;
        }
        curA = headA;
        curB = headB;
        if(lenB>lenA){
            swap(lenA,lenB);
            swap(curA,curB);
        }
        int gap = lenA - lenB;
        while(gap--){
            curA = curA->next;
        }
        while(curA!=NULL){
            if(curA==curB){
                return curA;
            }
            curA = curA->next;
            curB = curB->next;
        }
        return NULL;

    }
};
```


### 53、在排序数组中查找数字I

题目链接：https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/

C++:
```C++
class Solution {
public:
    int binarySearch(vector<int>& nums, int target, bool lower){
        int left = 0;
        int right = nums.size()-1;
        int res = nums.size();
        while(left<=right){
            int mid = left + (right-left)/2;
            if(nums[mid]>target||(lower && nums[mid]>=target)){
                right = mid-1;
                res = mid;
            }else{
                left = mid+1;
            }
        }
        return res;
    }
    int search(vector<int>& nums, int target) {
        int leftIndex = binarySearch(nums,target,true);
        int rightIndex = binarySearch(nums,target,false)-1;
        if(leftIndex<=rightIndex && rightIndex<=nums.size() && nums[leftIndex]==target
            && nums[rightIndex]==target){
                return rightIndex-leftIndex+1;
            }
        return 0;
    }
};
```
Python:
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        def binarySearch(nums,target,lower):
            left=0
            right=len(nums)-1
            res=len(nums)
            while left<=right:
                mid=left+(right-left)//2
                if nums[mid]>target or (lower and nums[mid]>=target):
                    right=mid-1
                    res=mid
                else:
                    left=mid+1
            return res
        
        leftIndex=binarySearch(nums,target,True)
        rightIndex=binarySearch(nums,target,False)-1
        if leftIndex>=0 and rightIndex<=len(nums) and leftIndex<=rightIndex and nums[leftIndex]==target\
            and nums[rightIndex]==target:
            return rightIndex-leftIndex+1
        return 0
```


### 54、0~n-1中缺失的数字

题目链接：https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/

C++:
```C++
class Solution {
public:
    int missingNumber(vector<int>& nums) {
      if(nums.size()<=0) return -1;
      int left=0;
      int right=nums.size()-1;
      while(left<=right){
          int mid = left+(right-left)/2;
          if(nums[mid]!=mid){
              if(mid==0 || nums[mid-1]==mid-1){
                  return mid;
              }
              right = mid-1;
          }else{
              left = mid+1;
          }
      }
      if(left==nums.size()){
          return nums.size();
      }
      return -1;
    }
};
```
Python：
```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        if len(nums)<=0:
            return -1
        
        left=0
        right=len(nums)-1
        while left<=right:
            mid = left+(right-left)//2
            if nums[mid]!=mid:
                if mid==0 or nums[mid-1]==mid-1:
                    return mid
                right = mid - 1
            else:
                left = mid + 1
        if left==len(nums):
            return len(nums)
        return -1
```


### 55、二叉树搜索树的第k大节点

题目链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/

C++:
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int res;
    int kthLargest(TreeNode* root, int k) {
        if(root==NULL||k<0) return 0;
        dfs(root,k);
        return res;
    }
    //按照中序遍历倒序的方式，右->中->左，得到一个倒序排列的数组
    void dfs(TreeNode* root, int& k){
        if(root==NULL) return;
        dfs(root->right,k);
        k--;
        if(k==0) res = root->val;
        dfs(root->left,k);

    }
};
```


### 56、二叉树的深度

题目链接：https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/

C++:
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
       if(root==NULL) return 0;
       return 1+max(maxDepth(root->left),maxDepth(root->right));
    }
};
```

### 57、平衡二叉树

题目链接：https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/

C++:
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int heightNum(TreeNode* node){
        if(node==NULL) return 0;
        return 1+max(heightNum(node->left),heightNum(node->right));
    }
    bool isBalanced(TreeNode* root) {
        if(root==NULL) return true;
        if(abs(heightNum(root->left)-heightNum(root->right))>1) return false;
        return isBalanced(root->left) && isBalanced(root->right);
      
    }
};
```


### 58、数组中数字出现的次数

题目链接：https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/

C++:
```C++
class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
      int res = 0;
      //1、先对所有数字进行一次异或，得到两个出现一次的数字的异或值
      for(int i=0;i<nums.size();i++){
          res ^= nums[i];
      }
      //2、在异或结果中找到任意为1的位
      int div = 1;
      while((div & res) == 0){
          div<<=1;
      }
      //3、根据这一位对所有数字进行分组
      int a=0,b=0;
      for(int j=0;j<nums.size();j++){
          //在每个组内进行异或，得到两个数字
          if(div&nums[j]){
              a ^= nums[j];
          }else{
              b ^= nums[j];
          }
      }
      return vector<int>{a,b};
    }
};
```
Python:
```python
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        res = 0
        for n in nums:
            res^=n
        div=1
        while div&res == 0:
            div<<=1
        a,b=0,0
        for n in nums:
            if div&n:
                a^=n
            else:
                b^=n
        return [a,b]
```


### 59、数组中数字出现的次数II

题目链接：https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/

C++:
```C++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        if(nums.size()<=0) return 0;
        int bits[32] = {0};
        for(int i = 0;i<nums.size();i++){
            int j = 0;
            //得到各个二进制为1的有多少个
            while(nums[i]){
                bits[j] += nums[i] %2 ;
                nums[i] /= 2;
                j++;
            }
        }

        int ans = 0;
        for(int i = 0;i < 32;i++){
            //利用%3求得到对应位置上有没有1，有的话乘于2的i次方
            ans += (1<<i)*(bits[i]%3);
        }
        return ans;

    }
};
```
Python:
```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        bits=[0]*32
        for i in range(len(nums)):
            for j in range(32):
                bits[j] += nums[i]&1
                nums[i] >>= 1
        
        ans = 0
        for i in range(32):
            ans += (1<<i)*(bits[i]%3)

        return ans if bits[31]%3==0 else ~(ans ^ 0xffffffff)
```


### 60、和为s的两个数字

题目链接：https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/

C++:
```C++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
     if(nums.size()<=1) return vector<int>{0,0};
     int left = 0;
     int right = nums.size()-1;
     while(left<right){
         int Sum = nums[left]+nums[right];
         if(Sum==target){
             return vector<int>{nums[left],nums[right]};
         }else if(Sum>target){
             right--;
         }else{
             left++;
         }
     }
     return vector<int>{0,0};
    }
};
```


### 61、和为s的连续正数序列

题目链接：https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/

C++:
```C++
class Solution {
public:
    vector<int> genVec(int left, int right){
        vector<int> vec;
        for(int i=left;i<=right;i++){
            vec.push_back(i);
        }
        return vec;
    }
    vector<vector<int>> findContinuousSequence(int target) {
      vector<vector<int>> res;
      if(target<3) return res;

      int left = 1;
      int right = 2;
      int mid = (1+target)/2;
      int Sum = left+right;

      while(left<mid){
          if(Sum==target){
             res.push_back(genVec(left,right));
          }

          while(Sum>target && left<mid){
              Sum -= left;
              left++;
              if(Sum==target){
                  res.push_back(genVec(left,right));
              }
          }

          right++;
          Sum += right;
      }
      return res;
    }
};
```


### 62、翻转单词顺序

题目链接：https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/

C++:
```C++
class Solution {
public:
    string reverseWords(string s) {
        if(s.empty()) return "";
        s += " ";
        string temp = "";
        vector<string> res;
        for(char ch: s){
            if(ch==' '){
                if(!temp.empty()){
                    res.push_back(temp);
                    temp.clear();
                }
            }else{
                temp += ch;
            }
        }
        string ans = "";
        reverse(res.begin(),res.end());
        for(string &str: res){
            ans += str + ' ';
        }
        ans.pop_back();//用于删除最末尾的空格
        return ans;
    }
};
```

### 63、左旋转字符串

题目链接：https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/

C++:
```C++
class Solution {
public:
    string reverseLeftWords(string s, int n) {
      reverse(s.begin(),s.begin()+n);
      reverse(s.begin()+n,s.end());
      reverse(s.begin(),s.end());
      return s;
    }
};
```


### 64、滑动窗口的最大值

题目链接：https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/

C++:
```C++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> maxInWindows;
        if(nums.size()>=k && k>=1){
            deque<int> que;
            for(int i=0;i<k;i++){
                while(!que.empty() && nums[i]>=nums[que.back()]){
                    que.pop_back();
                }
                que.push_back(i);
            }

            for(int i=k;i<nums.size();i++){
                maxInWindows.push_back(nums[que.front()]);
                while(!que.empty() && nums[i]>=nums[que.back()]){
                    que.pop_back();
                }
                if(!que.empty() && que.front()<=(i-k)){
                    que.pop_front();
                }
                que.push_back(i);
            }
            maxInWindows.push_back(nums[que.front()]);
        }
        return maxInWindows;

    }
};
```

### 65、队列的最大值

题目链接：https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/

C++:
```C++
class MaxQueue {
public:
    MaxQueue() {

    }
    queue<int> que;
    deque<int> dqe;
    
    int max_value() {
       if(dqe.empty()) return -1;
       return dqe.front();
    }
    
    void push_back(int value) {
        while(!dqe.empty() && value>dqe.back()){
            dqe.pop_back();
        }
        dqe.push_back(value);
        que.push(value);

    }
    
    int pop_front() {
        if(que.empty()) return -1;
        int ans = que.front();
        if(ans==dqe.front()){
            dqe.pop_front();
        }
        que.pop();
        return ans;
    }
};

/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue* obj = new MaxQueue();
 * int param_1 = obj->max_value();
 * obj->push_back(value);
 * int param_3 = obj->pop_front();
 */
 ```


### 66、n个骰子的点数

题目链接：https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/

C++:
```C++
class Solution {
public:
    vector<double> dicesProbability(int n) {
      vector<double> dp(6,1.0/6.0);
      for(int i=2;i<=n;i++){
          vector<double> tmp(5*i+1,0);
          for(int j=0;j<dp.size();j++){
              for(int k=0;k<6;k++){
                  tmp[j+k] += dp[j]/6.0;
              }
          }
          dp = tmp;
      }
      return dp;
    }
};
```

### 67、扑克牌中的顺子

题目链接：https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/

C++:
```C++
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        if(nums.size()<1) return false;

        //1、数组排序
        sort(nums.begin(),nums.end());

        //2、统计数组中0的个数
        int numZero = 0;
        for(int i=0;i<nums.size() && nums[i]==0; i++){
            numZero++;
        }

        //3、统计数组中的间隔数目
        int numGap = 0;
        int left = numZero;
        int right = left+1;
        while(right<nums.size()){
            //两数相等，有对子，则不可能是顺子
            if(nums[left]==nums[right]){
                return false;
            }

            numGap += nums[right] - nums[left] -1;
            left = right;
            right++;
        }
        return (numGap>numZero)?false:true;

    }
};
```
Python：
```python
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        if len(nums)<1: return False
        nums.sort()
        numofZero = 0
        for i in range(len(nums)):
            if nums[i]==0:
                numofZero+=1
        
        numofGap = 0
        left = numofZero
        right = numofZero+1
        while right<len(nums):
            if nums[left]==nums[right]:
                return False
            
            numofGap += nums[right]-nums[left] - 1
            left = right
            right +=1
        
        return False if numofGap>numofZero else True
```


### 68、圆圈中最后剩下的数字

题目链接：https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/

C++:
```C++
class Solution {
public:
    int lastRemaining(int n, int m) {
        //数学公式推导+循环
        if(n<1 || m<1) return -1;
        int res = 0;
        for(int i=2;i<=n;i++){
            res = (res+m)%i;
        }
        return res;

    }
};
```

### 69、股票的最大利润

题目链接：https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/

C++:
```C++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
       if(prices.size()<2) return 0;
       int min = prices[0];
       int maxDiff = prices[1]-min;
       for(int i=2;i<prices.size();i++){
           if(prices[i-1]<min){
               min = prices[i-1];
           }
           int curDiff = prices[i]-min;
           if(curDiff>maxDiff){
               maxDiff = curDiff;
           }
       }
       return maxDiff<0 ? 0 : maxDiff;
    }
};
```
Python:
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices)<2: return 0
        minPrice = prices[0]
        maxDiff = prices[1] - minPrice
        for i in range(2,len(prices)):
            if prices[i-1]<minPrice:
                minPrice = prices[i-1]
            curDiff = prices[i]-minPrice
            if curDiff>maxDiff:
                maxDiff = curDiff
        
        return 0 if maxDiff<0 else maxDiff
```

### 70、求1+2+3+...+n

题目链接：https://leetcode-cn.com/problems/qiu-12n-lcof/

C++:
```c++
class Solution {
public:
//使用前N项和公式法
    int sumNums(int n) {
       return ((int)pow(n,2)+n)>>1;
    }

//递归法
    int sumNums(int n){
        int sum = n;
        bool ans = (n>0) && ((sum += sumNums(n-1))>0);
        return sum;
    }
};
```


### 71、不用加减乘除做加法

题目链接：https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/

C++:
```C++
class Solution {
public:
    int add(int a, int b) {
        while (b) {
            int carry = a & b; // 计算 进位
            a = a ^ b; // 计算 本位
            b = (unsigned)carry << 1;
        }
        return a;
    }
};
```
Python：
```python
class Solution:
    def add(self, a: int, b: int) -> int:
        x = 0xffffffff
        a, b = a & x, b & x
        while b != 0:
            a, b = (a ^ b), (a & b) << 1 & x
        return a if a <= 0x7fffffff else ~(a ^ x)
```


### 72、构建乘积数组

题目链接：https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/

C++:
```C++
class Solution {
public:
    vector<int> constructArr(vector<int>& a) {
        vector<int> B(a.size());
        if(a.size()<=0) return B;
        B[0] = 1;
        //计算上三角
        for(int i=1;i<a.size();i++){
            B[i] = B[i-1]*a[i-1];
        }
        //计算下三角
        int temp = 1;
        for(int i=a.size()-1;i>=0;i--){
            B[i] = temp*B[i];
            temp = a[i]*temp;
        }
        return B;

    }
};
```


### 73、把字符串转成整数

题目链接：https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/

C++:
```C++
class Solution {
public:
    int strToInt(string str) {
        int ans = 0;
        bool start = false;//判断数字是否开始出现
        bool flag = true;//记录正负号
        for(int i=0;i<str.size();i++){
            if(start==false){
                while(str[i]==' '){
                    i++;
                }
                if(str[i]=='+'){
                    flag = true;//为正数
                    start = true;
                }else if(str[i]=='-'){
                    flag = false;//为负数
                    start = true;
                }else if(str[i]>='0' && str[i]<='9'){
                    start = true;
                    i--;
                }else{
                    return 0;
                }
            }
            else{
                if(str[i]>='0' && str[i]<='9'){
                    int top = str[i]-'0';
                    //此处代码的含义在于处理越界情况
                    //设拼接结果为ans,当前位的数字为top,大数边界(INT_MAX/10)为2147483647 // 10 = 214748364
                    //则有两种情况执行 ans = ans *10 + top后会越界
                    //1、ans > INT_MAX/10
                    //2、ans==INT_MAX/10 且 top>7 （INT_MAX的最后一位为7，乘10之后加top,如果top>7,则会越界）
                    if((ans>INT_MAX/10)||(ans==INT_MAX/10 && top>7)){
                        return flag ? INT_MAX : INT_MIN;
                    }
                     ans = ans *10 +top;
                }else{
                    break;
                }
               
            }
        }
        return flag ? ans : 0-ans;

    }
};
```

### 74、二叉搜索树的最近公共祖先

题目链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/

C++:
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root->val > p->val && root->val > q->val){
            return lowestCommonAncestor(root->left,p,q);
        }else if(root->val < p->val && root->val < q->val){
            return lowestCommonAncestor(root->right,p,q);
        }else{
            return root;
        }
        
    }
};
```

### 75、二叉树的最近公共祖先

题目链接：https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/

C++:
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root==NULL || root==p || root==q) return root;
        TreeNode* left = lowestCommonAncestor(root->left,p,q);
        TreeNode* right = lowestCommonAncestor(root->right,p,q);
        if(left!=NULL && right!=NULL) return root;
        if(left==NULL) return right;
        return left;
    }
};
```

### 76、正则表达式匹配

题目链接：https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/

C++:
```C++
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.size();
        int n = p.size();

        auto matches = [&](int i,int j){
            if(i==0){
                return false;
            }
            if(p[j-1]=='.'){
                return true;
            }
            return s[i-1]==p[j-1];
        };

        vector<vector<int>> f(m+1,vector<int>(n+1));
        f[0][0]=true;
        for(int i=0;i<=m;i++){
            for(int j=1;j<=n;j++){
                if(p[j-1]=='*'){
                    f[i][j] |= f[i][j-2];
                    if(matches(i,j-1)){
                        f[i][j] |= f[i-1][j];
                    }
                }else{
                    if(matches(i,j)){
                        f[i][j] |= f[i-1][j-1];
                    }
                }
            }
        }
        return f[m][n];

    }
};
```

### 77、从上到下打印二叉树

题目链接：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/

C++:
```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> levelOrder(TreeNode* root) {
        vector<int> res;
        queue<TreeNode*> que;
        if(root!=NULL) que.push(root);
        while(!que.empty()){
            int size = que.size();
            for(int i=0;i<size;i++){
                TreeNode* node = que.front();
                res.push_back(node->val);
                que.pop();
                if(node->left) que.push(node->left);
                if(node->right) que.push(node->right);
            }
        }
        return res;

    }
};
```








 








 



 
























