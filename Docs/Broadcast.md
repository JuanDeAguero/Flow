# Broadcast
```cpp
NARRAY Broadcast( NARRAY arr, vector<int> shape );
```
Broadcasts the array <i>arr</i> to the shape <i>shape</i>.</br>
Broadcasting allows doing elementwise operations on arrays of different (but compatible) shapes.</br></br>
Two shapes are compatible if for each element of the shape with the smaller size, the element at the same index in the other shape is equal or (at least) one of the two elements is a 1.
Same applies if the two shapes have the same size.
### Example
```cpp
NARRAY arr = Create( { 3 }, { 1, 2, 3 } );
NARRAY broadcasted = Broadcast( arr, { 3, 3 } );
```
```bash
broadcasted:
{
  { 1, 2, 3 },
  { 1, 2, 3 },
  { 1, 2, 3 }
}
```
