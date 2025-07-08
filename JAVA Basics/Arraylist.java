import java.util.ArrayList;

public class Arraylist {
    public static void main(String[] args) {
        ArrayList<Integer> arr = new ArrayList<>();
        arr.add(10);
        arr.add(20);
        arr.add(30);
        arr.add(40);
        arr.add(50);
        arr.add(60);

        System.out.println(arr);

        System.out.println(arr.get(2));
        arr.set(1, 100);
        System.out.println(arr.get(1));
    }
}
/*
 * 
 * | Method | Description | Example |
 * | --------------------- | ---------------------------------- |
 * ----------------------------------- |
 * | `add(element)` | Adds element at the end | `list.add(10);` |
 * | `add(index, element)` | Inserts element at given index | `list.add(1, 20);`
 * |
 * | `get(index)` | Retrieves element at index | `list.get(0);` |
 * | `set(index, element)` | Updates element at index | `list.set(0, 100);` |
 * | `remove(index)` | Removes element at index | `list.remove(1);` |
 * | `remove(object)` | Removes first occurrence of object |
 * `list.remove(Integer.valueOf(10));` |
 * | `size()` | Returns number of elements | `list.size();` |
 * | `isEmpty()` | Checks if empty | `list.isEmpty();` |
 * | `clear()` | Removes all elements | `list.clear();` |
 * | `contains(object)` | Checks if element exists | `list.contains(20);` |
 * 
 */