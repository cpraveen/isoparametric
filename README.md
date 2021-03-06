An attempt to create a MappingFEField by reading a high order mesh in Gmsh format. At present it is able to read in the mesh. 

## Steps to run this code

Generate a grid, this is setup to generate a Q4 grid (5x5 points in each cell)

```
cd BL1_laminarJoukowskAirfoilGrids
python ./Laminar.py
cd ..
```

Compile and run

```
cmake .
make
./main
```

## Ordering of dofs in ```FE_Q```

```
-----------
Q1
-----------

*      0-------1
*  

*      2-------3
*      |       |
*      |       |
*      |       |
*      0-------1
*   

*         6-------7        6-------7
*        /|       |       /       /|
*       / |       |      /       / |
*      /  |       |     /       /  |
*     4   |       |    4-------5   |
*     |   2-------3    |       |   3
*     |  /       /     |       |  /
*     | /       /      |       | /
*     |/       /       |       |/
*     0-------1        0-------1
*  

-----------
Q2
-----------

*      0---2---1
* 

*      2---7---3
*      |       |
*      4   8   5
*      |       |
*      0---6---1
*  

*         6--15---7        6--15---7
*        /|       |       /       /|
*      12 |       19     12      1319
*      /  18      |     /       /  |
*     4   |       |    4---14--5   |
*     |   2---11--3    |       |   3
*     |  /       /     |      17  /
*    16 8       9     16       | 9
*     |/       /       |       |/
*     0---10--1        0---10--1
*
*         *-------*        *-------*
*        /|       |       /       /|
*       / |  23   |      /  25   / |
*      /  |       |     /       /  |
*     *   |       |    *-------*   |
*     |20 *-------*    |       |21 *
*     |  /       /     |   22  |  /
*     | /  24   /      |       | /
*     |/       /       |       |/
*     *-------*        *-------*
* 

-----------
Q3
-----------

*      0--2--3--1
*

*      2--10-11-3
*      |        |
*      5  14 15 7
*      |        |
*      4  12 13 6
*      |        |
*      0--8--9--1
* 

-----------
Q4
-----------

*      0--2--3--4--1
*

*      2--13-14-15-3
*      |           |
*      6  22 23 24 9
*      |           |
*      5  19 20 21 8
*      |           |
*      4  16 17 18 7
*      |           |
*      0--10-11-12-1
*   
```
