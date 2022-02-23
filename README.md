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


























