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














