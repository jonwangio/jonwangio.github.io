---
title: 'Data Structures and Algorithms'
note-type: Research
layout: note
head-title: 'Data Structures and Algorithms (Resarch)'
ext-js: https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML
---

This is a summary of common data structures and algorithms. It's intended to be used a terse summary that I can refer to for a very quick refresher. For a more graphical summary of just time and space complexity, check out the [Big O Cheatsheet](http://bigocheatsheet.com).

My implementation of these data structures and algorithms can be found at my [github common algorithms](https://github.com/hkhanna/common-algorithms) repository.

A more detailed set of things to know can be found at this [TripleByte Quip](/files/TripleByte_Quip.pdf).
## Data Structures

### Arrays

An array is a foundational data structure consisting of a collection of elements, each identified by at least one array index.

#### Read
- *All Cases*. You have the numerical index. With that, you can instantly look up an element. $$O(1)$$

#### Insert
- *Best Case*. Insert an element at the end of the array. You know the size so you know the address to tack an element on at the end. $$O(1)$$.
- *Worst Case*. Insert an element at the beginning of the array. You will need to scoot all $$N$$ elements down one spot to make room. $$O(N)$$.
- *Other Worst Case*. Insert an element at the end of the array, but there is no room left in memory. The all $$N$$ elements of the array need to be copied to a new location where memory is available. $$O(N)$$.

##### Amortized Time
Amortized time is the way to express the time complexity **when an algorithm has the very bad time complexity only once in a while besides the time complexity that happens most of time**. Good example would be an ArrayList which is a data structure that contains an array and can be extended. Another definition is **average time taken per operation, if you do many operations**.

Imagine that you want to insert N items into an initially *empty* array. And every time you want to insert an element until an array that is *already full*, it copies itself over to a new memory location with double the size (i.e., a *resizing factor* of 2). 

1. You insert 1. The empty array finds a new location of size 2, and inserts the 1.
2. You insert 2. The size-2 array is full, but no copying is needed at this time.
3. You insert 3. Before the insert, the size-2 array copies itself over to an array of size 4. It has to copy 2 elements over to do this. 
4. You insert 4. The size-4 array fills up, but no copying is needed at this time.
5. You insert 5. Before the insert, the size-4 array copies itself over to an array of size 8. It has to copy 4 elements over to do this. And so on.

The insertion step is $$O(1)$$ like normal. But whats the time complexity associated with all the copying steps? It's easier to think about it backwards. When the array reaches size N, the most recent copying step took N / 2 steps. And when the array reached size N / 2, the most recent copying step took N / 4. So the total number of copying steps took:

$$
N/2 + N/4 + N/8 + \dots + 2 + 1
$$

If you add up that series, you get N. And if the *total copying* steps took N time, and if there were N inserts, you divide N / N and get $$O(1)$$. So the copying steps, *on average* are $$O(1)$$. 

~~**What I don't understand**, though,~~ [*I get it now -- made a careless algebra error!*], is that it's highly unlikely that an array will be sized exactly to N. Rather, N items in an array will end up in an array of size N + X because the array kept doubling as you inserted $$1 \dots N$$. So the most recent copy was X / 2, which is somewhere between N / 2 and N. In the *worst case*, N is equal to $$\frac{X}{2} + 1$$, which means the the most recent copy was $$N - 1$$. So the total number of copying steps took:

$$
N - 1 + \frac{N - 1}{2} + \frac{N - 1}{4} + \dots + 2 + 1
$$

If you add up *this* series, you get $$(N - 1) + (N - 1) = 2N - 2$$. If there are N inserts and you divide $$\frac{2N - 2}{N}$$, ~~you get N minus a constant. So, isn't the average time for the copying steps O(N) in the worst case, *not* O(1)?~~ *you get $$O(1)$$*. 

#### Search
- *All Cases*. Go element-by-element until you find what you're looking for. Time complexity depends on where the element is, but because on average this takes $$\frac{N}{2}$$ steps, it is $$O(N)$$ complexity.

#### Delete
- Looks a lot like insert, except there is no chance you'll need to move the array due to size contraints (what I called the "*Other Worst Case*").
- *Best Case*. Delete element from the end of the array. $$O(1)$$.
- *Worst Case*. Delete element from beginning of the array. You need to scoot all $$N$$ elements over to fill the gap. $$O(N)$$.

### Sets

A set is an array with one additional constraint: no duplicates are allowed. Read, Search and Delete operations are the same as an array. 

But for Insert, you need to search the set first to prevent duplicates. So, best case takes N + 1 steps ($$O(N)$$). Worst case requires searching ($$N$$ steps), then scooting down all of the elements ($$N$$ steps), then inserting the element ($$1$$ step). This is $$2N + 1$$ steps, which is still just $$O(N)$$.

### Ordered Arrays

Same as unsorted arrays for Read and Delete. 

For Insert, you have to search to find the appropriate location first, which is on average $$O(N)$$. Then, you scoot down the rest of the items and finally insert. $$O(N)$$.

For Search, you can do a linear search, where you can stop when you get to the spot where the element should have been. It's still $$O(N)$$ but when the searched element is not in the array, it will almost always take less time than the unsorted array.

You can also do a **binary search**, where you keep dividing the array in half until you get to the spot where the element should be. This is $$O(log(n))$$. Every time we double the number of elements in the array, we only need to add *one more step*.

### Hash Tables

A hash table is a list of paired key-values. You look up a value by key. To do that, it hashes the key to a memory location and then pulls the value at that location in memory. 

If there are collisions, the classic approach called "separation chaining" works like this. The memory location at the hashed key points to an array of (key, value) pairs. It then needs to search through those (key, value) pairs to find the right key. 

A hash table's effiency depends on 3 factors:
- how many data is stored in the hash table
- how many cells are available in the has table
- which hash table function is being used
  
A good hash table strikes a balance of avoiding collisions without consuming lots of memory. The rule of thumb is that for every 7 data elements stored in the hash table, it should have ten cells. This ratio of data to cells is called the *load factor*. An ideal load factor, therefore, is 0.7.

#### Read
N/A. This is conceptually the same as search.

#### Insert
$$O(1)$$ all cases. Even if collision, you can just put the new item at the end of the array at the hash location.

#### Search
$$O(1)$$ normal case, $$O(N)$$ worst case if there are lots of collisions.

#### Delete
$$O(1)$$ normal case. If you're deleting from a collided / chained location, it could take up to $$O(N)$$.

### Stacks & Queues

Stacks and queues are just *arrays with restrictions*.

In a stack:
- data can only be inserted at the end (a/k/a the *top* of the stack)
- data can only be read from the end
- data can only be removed from the end

Stack is *LIFO*. 

In a queue:
- data can only be inserted at the end (identical to stack)
- data can only be read from the front (opposite of stack)
- data can only be removed from the front (opposite of stack)

Queue is *FIFO*.

### Linked List
> A node-based data structure

A linked list is the simplest of the node-based data structures. They've very similar to arrays in practice. Any application where you're using an array, you could also potentially use a linked list. 

Linked lists do not consist of a bunch of memory cells in a row, so you can't just index them at $$O(1)$$ like you can an array. Instead, they're spread across computer memory as nodes. Each node stores both the data and the location of the next node. The final node's link contains `null`. 

**Read**. You have to traverse the nodes from the first node to get to the one you're looking for. Best case, you're looking for the first node. You have that address, so this is $$O(1)$$. Worst case, you're trying to read the final node, so you need to traverse the whole list to read it, so this is $$O(N)$$. This is a disadvantage compared to arrays, which are $$O(1)$$ for read. 

**Search**. Same as read for linked lists -- you need to traverse the list to find what you're looking for. But arrays and linked lists have the same efficiency for search, $$O(N)$$. 

**Insert**. Insertion in a linked list *seems* like it should take 1 step since you just change a pointer and don't need to scoot the rest of the values down like in an array. But remember that you have to *traverse* the list before you can find the node you need to insert it at. So it takes, worst case, if you're inserting at the final node, N steps. Best case, you're inserting in the beginning, so it's 1 step. So it is $$O(N)$$. Although insertion in an array is $$O(N)$$ too, it is interestingly the exact opposite best case and worst case senarios. 

**Delete**. Same as insertion. Delete at the beginning takes 1 step. Delete at the end takes N steps because you have to traverse the list. So, $$O(N)$$. This is also the opposite of an array, which has swapped best case and worst case scenarios but is also technically $$O(N)$$.

Since insertion and deletion are a wash with arrays and reading from an array is faster than a linked list, why ever use a linked list? It really shines when you're doing many deletions or insertions at once while traversing the list. I.e., if you're going through a list of emails and deleting malformed once. Because you don't need to re-traverse with each deletion, each deletion becomes $$O(1)$$. 

#### Doubly Linked List

Each node stores its value and the address of both the next and the previous node. This makes inserting and deleting at the beginning and end of the list $$O(1)$$, although the middle and average case is still N / 2 steps, so $$O(N)$$. 

Doubly linked lists are very useful as a data structure for a *queue*, since those involve adding and removing from the ends of the data structure only.

### Binary Trees
> A node-based data structure

A *tree* is a node-based data structure that can have links to *multiple* nodes. The uppermost node is the *root* node. And nodes are *parents* and *children*. The root node is the parent node and not a child to any other node.

A binary tree is a tree in which each node has up to two children. 

A binary *search* tree is a binary tree that fits the specific ordering property of `all left descendents <= n < all right descendents` for all nodes n.

> These definitions aren't fixed, so good to clarify what assumptions hold.

A binary search tree is good when we want a data structure that maintains order and also has fast search, insertion and deletion. An ordered array is ordered but is slow on the options. A hash table is fast, but not ordered.

*Balanced* tree basically just means not terribly unbalanced. It's not a precise term. A balanced binary search tree should be rougly $$O(logN)$$ for it to be balanced.

*Complete* binary tree is one in which every level of the tree is fully filled except maybe the last level, and to the extent that the last level is filled, it is filled left to right.

*Full* binary tree is a tree in which every node has zero or two children (and no nodes have just one child).

*Perfect* binary trees are complete and full binary trees. Additionally, all leaf nodes will be at the same level and this level has the maximum number of nodes.

#### Searching

1. Start at the root node.
2. Inspect the value at the node.
3. If we've found the value, great.
4. If the value we're looking for is less than the current node, search for it in the left subtree.
5. If the value we're looking for is greater than the current node, search in the right subtree.

*Recursion* is very useful here.

Searching takes $$O(logN)$$ time, since you are constantly dividing the tree in half. But this is only best case, when the tree is perfectly-balanced. If it's unbalanced, it turns into $$O(N)$$ since the tree is only one sided and you need to traverse every node in search. Note that the best case is also the average case because *randomly* ordered data inserted into a binary search tree ends up close to perfectly balanced.

So, in a best case / average case scenario, searching is the same as binary search in an ordered array. 

#### Insertion

Here's where binary search trees really shine. 
1. Find the correct node to attach the new node to. Essentially a search.
2. Insert at the bottom of the tree.

This takes one extra step beyond a search. So, best case / average case $$O(logN)$$. Worst case would still be $$O(N)$$ I guess.

The key is to *randomize* your data before you insert it into a binary search tree. This is because *sorted* data ends up perfectly unbalanced based on this insertion algorithm. But *randomized* data ends up perfectly balanced. 

This is what makes binary search trees so much better than ordered arrays. Search and insert are both $$O(logN)$$ in the average case, but ordered arrays are $$O(N)$$ for insert.

#### Deletion
This is more complex.

1. If the node being deleted has no children, just delete it.
2. If the node being deleted has 1 child, delete it and plug the child into the spot where the deleted node was.
3. If the node has two children, replace the deleted node with the **successor node**. The successor node is the descendant node whose value is the least of all values that are greater than the deleted node, i.e., the "next number up" from the deleted value.
  - To get the successor node, visit the right child of the deleted node and then keep visiting the left child of each subsequent child until there are no more left children. The bottom value is the successor node.
4. **Edge case**: If the successor node has a right child, take that right child and turn it into the *left child of the former parent of the successor node*.  

Deletion is just search plus a few extra steps, so best case / average case is $$O(logN)$$. Contrast with an ordered array, for which deletion is $$O(N)$$. 

#### Tree Traversal

If you want to print every node in order, you can do an *inorder traversal*.

Recursion is a great tool for this:
1. Create a recursive function called that can be called on a particular node like the root note. The function then does this:
2. Call itself on the node's left child if it has one.
3. Visit the node. Do what you need, like printing.
4. Call itself on the node's right child if it has one.

Tree traversal has to visit every node so it is $$O(N)$$. 

Pre-order traversal visits the current node before its child nodes. The root node is always the first node visited.

Post-order traversal visits the current node after its child nodes. The root node is always the last node visited.

### Binary Heaps
A min-heap is a complete binary tree (not a binary search tree) where each node is smaller than its children. The root element is therefore the minimum element in the tree.

Max-heaps are just min-heaps where the elements are in descending rather than ascending order.

#### Insert
1. Insert the element at the bottom, rightmost spot to maintain completeness.
2. Fix the tree by swapping the new element with its parent until its in the right spot, i.e., bubble up the element.
   
This takes $$O(logN)$$

#### Extract Minimum Element
1. The minumum element is always at the top.
2. Remove it and swap it with the last element in the heap (the bottommost, rightmost).
3. Bubble down this element, swapping it with one of its children until the min-heap property is restored.
4. Do you swap it with the left child or right child? Since there's no inherent ordering between the two, swap it with the smaller one in order to maintain the min-heap ordering.

### Tries (Prefix Trees)
A trie is an n-ary tree in which characters are stored at each node. The * symbol represents the end of the path. A node in a trie can have anywhere from 1  to ALPHABET_SIZE + 1 children.

### Graphs
> A node-based data structure

In graph jargon, each node is called a *vertex* and each line is called an *edge*. Vertices that are connected by an edge are said to be *adjacent* to each other. Graphs can be directed (one-way) or undirected (two-way). A *connected graph* is where there are no isolated subgraphs. I.e., there is some path between every vertex. An *acyclic graph* is a graph without cycles.

Graphs can be implemented in a number of ways, but one of the simplest ways is using a hash table. The key is the node, the value is a list of the other nodes to which it is connected. You could also do an adjacency list or adjacency matrix. 

**Breadth first** should be used if we want to find the shortest path between two nodes.
**Depth first** should be used if we want to visit every node in the graph. Both would work find here, but DFS would be simpler.

#### Breadth-First Search
This uses a queue. Common pitfall is that it is *not recursive*. 

At the beginning the queue contains only our starting vertex. Then:
1. Visit each vertex adjacent to the current vertex. If it has not yet been visited, mark it as visited, and add it to the queue (but do not make it the current vertex!).
2. If the current vertex has no unvisited vertices adjacent to it, remove the next vertex from the queue and make it the current vertex.
3. If there are no more unvisited vertices adjacent to the current vertex, and there are no more vertices in the queue, the algorithm is complete.

There are V removals from the queue, and 2E visiting steps (since each edge is connected to 2 vertices). So a BFS has efficiency of $$O(V + E)$$.

#### Depth-First Search
*Recursion* is useful here.

In DFS, we visit a node `a` and the iterate through each of `a`'s neighbors. When visiting a node `b` that is a neighbor of `a`, we visit all of `b`'s neighbhors before going on to `a`'s other neighbhors. 

Pre-order and other forms of tree traveral are a form of DFS, but the key difference is that when implementing this for a graph, we must check if the node has been visited to avoid an infinite loop.

#### Dijkstra's Algorithm

This is one way of solving the "shortest path problem" when you have a weighted graph. Think, how to hack together the cheapest flights from a starting point (like, Atlanta) by flying through different airports.

## Algorithms

This is a good [sorting algorithm cheat sheet](https://www.interviewcake.com/sorting-algorithm-cheat-sheet).

### Thoughts on Big O Notation

- Big O usually describes the *worst case* scenario. 
- $$O(1)$$ is referred to as *constant time*. 
- $$O(1)$$ is the way to describe any algorithm that does not change its number of steps even when the data increases.
- Even a constant 100-step algorithm would be $$O(1)$$.
- $$O(N^2)$$ is the efficiency of algorithms when nested loops are used. When you see a nested loop, think $$O(N^2)$$.
- It is the *long-term growth rate* of algorithms. 
- Big O is useful for contrasting algorithms that fall under different classifications of Big O. When two algorithms falls under the *same* classification, further analysis is required.

#### Logarithm Time

- $$O(log(N))$$ describes an algorithm that increases *one step each time the data is doubled*.
- $$\log_2 8$$ means how many times do you have to multiply 2 by itself to get 8? (The answer is 3.)
- Another way of explaining $$\log_2 8$$ is if we kept dividing 8 by 2 until we ended up with 1, how many 2s would we have in our equation?
- Said simply: $$O(log(N))$$ means that the algorithm takes as many steps as it takes to keep halving the data elements until we remain with one. 

### Space Complexity
> Big O as applied to memory, rather than time.

For N elements of data, an algorithm consumes a relative number of *additional* units in memory. Some people thing of space complexity as including the original data structure. You should be clear when you're talking about it. 

### Triangular Series

It's helpful to know that:

$$N + (N - 1) + (N - 2) + (N - 3) + \dots + 1$$ 

is equal to:

$$
\frac{N^2 + N}{2}
$$

This is the so-called **triangular series**.

### Bubble Sort
*Given an array of unsorted numbers, how can we sort them so that they end up in ascending order?*

1. Pass through the array and at each step, swap the elements if they are out of order.
2. This will "bubble up" the highest number to the end of the array. 
3. Repeat. In each passthrough, the highest unsorted value bubbles up to its correct position.
4. Stop when there is a round where there are no swaps. 

In each passthrough, you can stop one element earlier since you know the ones at the top are already in order. For N, elements, you therefore make 

$$(N - 1) + (N - 2) + (N - 3) + \dots + 1$$ 

comparisons. In other words,

$$\sum_{i=1}^{N-1} N - i$$

This is very close to the triangular series. In fact, it's the triangular series minus N.

$$
\frac{N^2 - N}{2}
$$

If the array is sorted the exactly wrong way so that you need to make a swap for every comparison, you just multiply that by 2, which gets it even closer to $$N^2$$ time. So, best case is 
$$
\frac{N^2 - N}{2}
$$
and worst case is 
$$
N^2 - N
$$
.

### Searching for Duplicates in an Array

- The brute-force solution is a nested loop. But this is $$O(N^2)$$. 
- You can also just keep a tracker array and have a single loop, which is $$O(N)$$ but also has space complexity of $$O(N)$$.

### Selection Sort

Selection sort is basically bubble sort, but in reverse and with only 1 swap per passthrough rather than potentially N swaps per passthrough.

1. On each passthrough, start at the next index. So if on passthrough 1, you start at index 0; on passthrough 2, you start at index 1.
1. Pass through the array and at each step, store the index of the lowest value.
1. Once you have the index of the lowest value, swap it with the index that you started on.
1. Continue until you've completed N passthroughs and all the data will be sorted. 

This is just best case complexity for bubble sort:
$$
\frac{N^2 - N}{2}
$$

And there is no worst case. That's all the cases.

### Insertion Sort

Insertion sort works by inserting elements from an unsorted list into a sorted subsection of the list, one item at a time.
Insertion sort reveals the power of analyzing scenarios beyond the worst case. 

1. On each passthrough, start at the next index. So if on passthrough 1, you start at index 1; on passthrough 2, you start at index 2.
1. On the first pass, temporarily remove the value at index 1 (the second cell), and store it in a temp variable.
1. This leaves a gap at index 1. 
1. Compare each value to the left of the gap with the temp varaible. If it's greater than the temp variable, shift the value to the right. 
1. If it's less than the temp variable, stop. Insert the temp variable in the gap.
1. Repeat, starting at the next index. 

To analyze the complexity, there are four types of steps that occur in insertion sort: removals, comparisons, shifts and insertions. 

- For removals: there is exactly 1 removal for each of N - 1 passthroughs, i.e., N - 1 steps.
- For insertions: there is exactly 1 insertion for each of N - 1 passthroughts, i.e., N - 1 steps.
- For comparisons:
  - In a best case scenario where the array is already sorted, there is 1 comparison for each passthrough. On that comparison, the compared value is less than the temp variable, so it stops there.  That's N - 1 steps.
  - In a worst case senario, there's a triangular series number of comparisons, which is $$\frac{N^2 - N}{2}$$. 
- For shifts, in a best case senario there are no shifts. In a worst case senario, it's the triangular series. 

Added all up, the best case scenario looks like:

$$(N - 1) + (N - 1) + (N - 1) + 0 = 3N - 3$$

Worst case senario looks like:

$$(N - 1) + (N - 1) + \frac{N^2 - N}{2} + \frac{N^2 - N}{2} = N^2 + N - 2$$ 

In the average case senario, there are on average half the number of comparisons and shifts as the worst case scenario. So it looks like:

$$(N - 1) + (N - 1) + \frac{N^2 - N}{4} + \frac{N^2 - N}{4} = \frac{N^2 + 3N - 4}{2}$$

So, whether you use Selection Sort or Insertion Sort, which are both technically $$O(N^2)$$ complexity, depends on which case you expect. For a mostly-sorted, best case, use Insertion Sort, which will perform close to $$O(N)$$. For an average case, they perform similarly. For worst case, Selection Sort will perform best.

### Quicksort

In many languages, the sorting algo performed under the hood is Quicksort. In average case, it's $$O(NlogN)$$. In worst case, it's $$O(N^2)$$.

#### Partitioning

To partition an array, take a random value from the array which will be called the *pivot*. Make sure every number that is less than the pivot ends up to the left of the pivot and that every number greater than the pivot end up to the right of the pivot. 

1. Select a pivot. By convention, this is always the right-most value.
2. Assign "pointers" -- one to the left-most value of the array, one to the right-most value of the array, excluding the pivot itself.
3. The left pointer continuously moves one cell to the right until it reaches a value greater than or equal to the pivot. Then it stops.
4. Then, the right pointer continuously moves one cell to the left until it reaches a value that is less than or equal to the pivot. Then stops.
5. Swap the values that the left and right pointers are pointing to.
6. Repeat steps 3 and 4  until the points are pointing to the same vlue or the left pointer is to the right of the right pointer.
7. Swap the pivor with the left pointer value. 

Once this is done, all the values to the left of the pivot are less than the pivot and all the values to the right of the pivot are greater than the pivot. The pivot is therefore in the correct position in the array, although the other values are not necessarily completely sorted yet. 

#### Quicksort Algorithm

1. Partition the array. The first pivot is now in its proper place.
2. Treat the subarrays to the left and right of pivot as their own arrays.
3. Repeat step 1 and 2 recursively for all subarrays until we have a subarray with 0 or 1 element. That's the base case and do nothing.


#### Quicksort Efficiency

Each *partition* consists of two types of steps:
- comparisons: compare each value to the pivot. Each partition has N comparisons since the right and left pointers move together until they meet, so exactly all elements in the partition get evaluated.
- swaps: swap the values being pointed to by the left and right pointers. Best case, 0 or 1 swap (just the pivot). Worst case, N / 2 swaps, since every value at the left pointer will be swaped with every value at the right pointer. Average case is half of worst case, i.e., N / 4 swaps. 

A partition therefore runs in $$O(N)$$ time. 

In the average case, the pivot gets moved to the center. This halves N with every iteration. Because we're halving the N with every iteration of the partition, this is a lot like binary search, which is in $$O(logN)$$ time. And since there are N pivots, quicksort runs, in the average case, at $$O(NlogN)$$ time.

In the worst case, the pivot doesn't get swapped into the center but stays to the one side. This makes worst case quicksort operate at $$O(N^2)$$ time. 

### Merge Sort

*Recursion* is useful here.

Divide the array in half, sort each of those halves, then merge them back together. Each of the halves has the same sorting algorithm applied to it. Eventually you are just merging just two single-element arrays. 

The "merge" function does the heavy lifting. To merge two halves, you compare the left-most element on the left half with the left-most element on the right and store the lesser one in the left-most spot in the new, merged array. Then you move one spot to the right on the array half that you just copied from and do it again. I.e., you compare the earliest unmerged items from both lists and add the smaller one. 

This article on [merge sort](https://www.interviewcake.com/concept/python/merge-sort) explains it better.

Time complexity is $$O(NlogN)$$ because each comparison step takes N steps, and there are logN splits since you are splitting in half.

Space complexity is $$O(N)$$ if you do it efficiently, but $$O(NlogN)$$ if you do it less thoughtfully.

### Radix Sort

### Quickselect

Quickselect allows you to choose a certain value, like the tenth-lowest value in the array. You don't need to sort the whole thing. 

Quickselect relies on partitioning just like quicksort and can be thought of as a hybrid quicksort and binary search. Its efficiency is $$O(N)$$ for average scenarios. It's not as efficient as binary search because although it's halving every time, it has to do all the pivoting in that half. It works out in the end to $$O(N)$$, although I admit the intuition is still fuzzy to me.

### Recursion

Recursion is a natural fit in any situation where you find yourself having to repeat an algorithm within the same algorithm.

The case in which the method will *not* recurse is the **base case**. 

One approach to reading recursive code is:
1. Identify the base case
2. Walk through the function assuming it's the base case
3. Then, walk through the function its dealing with the case immediately before the base case.
4. Progress by moving up the cases on at a time. 

When you have a recursive function that makes multiple calls, the runtime will often (but not always) look like $$O(branches^{depth})$$, where branches is the number of times each recursive call branches.

All recursive functions *can* be implemented iteratively, although they may be much more complex.
