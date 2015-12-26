#include <deal.II/base/utilities.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_handler.h>

#include <fstream>

#define dim       2
#define spacedim  2

using namespace dealii;

/* Some exceptions needed by read_msh */
DeclException0 (ExcNoTriangulationSelected);

DeclException2 (ExcInvalidVertexIndex,
                int, int,
                << "While creating cell " << arg1
                << ", you are referencing a vertex with index " << arg2
                << " but no vertex with this index has been described in the  input file.");

DeclException1 (ExcInvalidGMSHInput,
                std::string,
                << "The string <" << arg1 << "> is not recognized at the present"
                << " position of a Gmsh Mesh file.");

DeclException1 (ExcGmshUnsupportedGeometry,
                int,
                << "The Element Identifier <" << arg1 << "> is not "
                << "supported in the deal.II library when "
                << "reading meshes in " << dim << " dimensions.\n"
                << "Supported elements are: \n"
                << "ELM-TYPE\n"
                << "1 Line (2 nodes, 1 edge).\n"
                << "3 Quadrilateral (4 nodes, 4 edges).\n"
                << "5 Hexahedron (8 nodes, 12 edges, 6 faces) when in 3d.\n"
                << "15 Point (1 node, ignored when read)");


DeclException0 (ExcGmshNoCellInformation);
//---------------------------------------------------------------------------
// Used to store all vertex coordinates and for each cell, the vertices
// giving defining the isoparametric information.
//---------------------------------------------------------------------------
struct GridData
{
public:
   unsigned int n_vertices;
   std::vector<double> x, y, z;
   std::vector< std::vector<unsigned int> > v_in_cell;
};

//---------------------------------------------------------------------------
// This function taken from v8.3.0 and modified.
// It works for Q1 to Q4 elements in 1d and 2d.
//---------------------------------------------------------------------------
void read_msh (std::istream &in, Triangulation<dim> *tria, GridData &grid_data)
{
   Assert (tria != 0, ExcNoTriangulationSelected());
   AssertThrow (in, ExcIO());
   
   unsigned int n_vertices;
   unsigned int n_cells;
   unsigned int dummy;
   std::string line;
   
   in >> line;
   
   // first determine file format
   unsigned int gmsh_file_format = 0;
   if (line == "$NOD")
      gmsh_file_format = 1;
   else if (line == "$MeshFormat")
      gmsh_file_format = 2;
   else
      AssertThrow (false, ExcInvalidGMSHInput(line));
   
   // if file format is 2 or greater
   // then we also have to read the
   // rest of the header
   if (gmsh_file_format == 2)
   {
      double version;
      unsigned int file_type, data_size;
      
      in >> version >> file_type >> data_size;
      
      Assert ( (version >= 2.0) &&
              (version <= 2.2), ExcNotImplemented());
      Assert (file_type == 0, ExcNotImplemented());
      Assert (data_size == sizeof(double), ExcNotImplemented());
      
      // read the end of the header
      // and the first line of the
      // nodes description to synch
      // ourselves with the format 1
      // handling above
      in >> line;
      AssertThrow (line == "$EndMeshFormat",
                   ExcInvalidGMSHInput(line));
      
      in >> line;
      // if the next block is of kind
      // $PhysicalNames, ignore it
      if (line == "$PhysicalNames")
      {
         do
         {
            in >> line;
         }
         while (line != "$EndPhysicalNames");
         in >> line;
      }
      
      // but the next thing should,
      // in any case, be the list of
      // nodes:
      AssertThrow (line == "$Nodes",
                   ExcInvalidGMSHInput(line));
   }
   
   // now read the nodes list
   in >> n_vertices;
   std::vector<Point<spacedim> >     vertices (n_vertices);
   // set up mapping between numbering
   // in msh-file (nod) and in the
   // vertices vector
   std::map<int,int> vertex_indices;
   grid_data.x.resize (n_vertices);
   grid_data.y.resize (n_vertices);
   grid_data.z.resize (n_vertices);
   
   for (unsigned int vertex=0; vertex<n_vertices; ++vertex)
   {
      int vertex_number;
      double x[3];
      
      // read vertex
      in >> vertex_number
      >> x[0] >> x[1] >> x[2];
      
      for (unsigned int d=0; d<spacedim; ++d)
         vertices[vertex](d) = x[d];
      // store mapping
      vertex_indices[vertex_number] = vertex;
      
      grid_data.x[vertex] = x[0];
      grid_data.y[vertex] = x[1];
      grid_data.z[vertex] = x[2];
   }
   
   // Assert we reached the end of the block
   in >> line;
   static const std::string end_nodes_marker[] = {"$ENDNOD", "$EndNodes" };
   AssertThrow (line==end_nodes_marker[gmsh_file_format-1],
                ExcInvalidGMSHInput(line));
   
   // Now read in next bit
   in >> line;
   static const std::string begin_elements_marker[] = {"$ELM", "$Elements" };
   AssertThrow (line==begin_elements_marker[gmsh_file_format-1],
                ExcInvalidGMSHInput(line));
   
   in >> n_cells;
   
   // set up array of cells
   std::vector<CellData<dim> > cells;
   SubCellData                 subcelldata;
   
   for (unsigned int cell=0; cell<n_cells; ++cell)
   {
      // note that since in the input
      // file we found the number of
      // cells at the top, there
      // should still be input here,
      // so check this:
      AssertThrow (in, ExcIO());
      
      unsigned int cell_type;
      unsigned int material_id;
      unsigned int nod_num;
      
      /*
       For file format version 1, the format of each cell is as follows:
       elm-number elm-type reg-phys reg-elem number-of-nodes node-number-list
       
       However, for version 2, the format reads like this:
       elm-number elm-type number-of-tags < tag > ... node-number-list
       
       In the following, we will ignore the element number (we simply enumerate
       them in the order in which we read them, and we will take reg-phys
       (version 1) or the first tag (version 2, if any tag is given at all) as
       material id.
       */
      
      in >> dummy          // ELM-NUMBER
      >> cell_type;     // ELM-TYPE
      
      switch (gmsh_file_format)
      {
         case 1:
         {
            in >> material_id  // REG-PHYS
            >> dummy        // reg_elm
            >> nod_num;
            break;
         }
            
         case 2:
         {
            // read the tags; ignore
            // all but the first one
            unsigned int n_tags;
            in >> n_tags;
            if (n_tags > 0)
               in >> material_id;
            else
               material_id = 0;
            
            for (unsigned int i=1; i<n_tags; ++i)
               in >> dummy;
            
            nod_num = GeometryInfo<dim>::vertices_per_cell;
            
            break;
         }
            
         default:
            AssertThrow (false, ExcNotImplemented());
      }
      
      
      /*       `ELM-TYPE'
       defines the geometrical type of the N-th element:
       `1'
       Line (2 nodes, 1 edge).
       
       `3'
       Quadrangle (4 nodes, 4 edges).
       
       `5'
       Hexahedron (8 nodes, 12 edges, 6 faces).
       
       `15'
       Point (1 node).
       */
      
      // how many vertices does this element have
      unsigned int nv;
      switch(cell_type)
      {
         // single point
         case 15:
            nv = 1;
            break;
         // Line elements
         case 1:
            nv = 2;
            break;
         case 8:
            nv = 3;
            break;
         case 26:
            nv = 4;
            break;
         case 27:
            nv = 5;
            break;
         // quadrangle elements
         case 3:
            nv = 4;
            break;
         case 10:
            nv = 9;
            break;
         case 36:
            nv = 16;
            break;
         case 37:
            nv = 25;
            break;
         default:
            AssertThrow(false, ExcNotImplemented());
      }
      
      if (((cell_type==1 || cell_type==8 || cell_type==26 || cell_type==27) && (dim == 1)) ||
          ((cell_type==3 || cell_type==10 || cell_type==37) && (dim == 2)) ||
          ((cell_type == 5) && (dim == 3)))
         // found a cell
      {
         AssertThrow (nod_num == GeometryInfo<dim>::vertices_per_cell,
                      ExcMessage ("Number of nodes does not coincide with the "
                                  "number required for this object"));
         
         std::vector<unsigned int> all_verts(nv);
         
         // allocate and read indices
         cells.push_back (CellData<dim>());
         for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
         {
            in >> cells.back().vertices[i];
            all_verts[i] = cells.back().vertices[i] - 1;
         }
         
         // read remaining points so we can put them into grid_data
         for(unsigned int i=GeometryInfo<dim>::vertices_per_cell; i<nv; ++i)
         {
            in >> all_verts[i];
            --all_verts[i];
         }
         
         grid_data.v_in_cell.push_back (all_verts);
         
         // to make sure that the cast wont fail
         Assert(material_id<= std::numeric_limits<types::material_id>::max(),
                ExcIndexRange(material_id,0,std::numeric_limits<types::material_id>::max()));
         // we use only material_ids in the range from 0 to numbers::invalid_material_id-1
         Assert(material_id < numbers::invalid_material_id,
                ExcIndexRange(material_id,0,numbers::invalid_material_id));
         
         cells.back().material_id = static_cast<types::material_id>(material_id);
         
         // transform from ucd to
         // consecutive numbering
         for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
         {
            AssertThrow (vertex_indices.find (cells.back().vertices[i]) !=
                         vertex_indices.end(),
                         ExcInvalidVertexIndex(cell, cells.back().vertices[i]));
            
            // vertex with this index exists
            cells.back().vertices[i] = vertex_indices[cells.back().vertices[i]];
         }
      }
      else if ((cell_type==1 || cell_type==8 || cell_type==26 || cell_type==27)
               && ((dim == 2) || (dim == 3)))
         // boundary info
      {
         subcelldata.boundary_lines.push_back (CellData<1>());
         in >> subcelldata.boundary_lines.back().vertices[0]
         >> subcelldata.boundary_lines.back().vertices[1];
         
         // skip remaining points
         double idummy;
         for(unsigned int i=2; i<nv; ++i)
            in >> idummy;
         
         // to make sure that the cast wont fail
         Assert(material_id<= std::numeric_limits<types::boundary_id>::max(),
                ExcIndexRange(material_id,0,std::numeric_limits<types::boundary_id>::max()));
         // we use only boundary_ids in the range from 0 to numbers::internal_face_boundary_id-1
         Assert(material_id < numbers::internal_face_boundary_id,
                ExcIndexRange(material_id,0,numbers::internal_face_boundary_id));
         
         subcelldata.boundary_lines.back().boundary_id
         = static_cast<types::boundary_id>(material_id);
         
         // transform from ucd to
         // consecutive numbering
         for (unsigned int i=0; i<2; ++i)
            if (vertex_indices.find (subcelldata.boundary_lines.back().vertices[i]) !=
                vertex_indices.end())
               // vertex with this index exists
               subcelldata.boundary_lines.back().vertices[i]
               = vertex_indices[subcelldata.boundary_lines.back().vertices[i]];
            else
            {
               // no such vertex index
               AssertThrow (false,
                            ExcInvalidVertexIndex(cell,
                                                  subcelldata.boundary_lines.back().vertices[i]));
               subcelldata.boundary_lines.back().vertices[i] =
               numbers::invalid_unsigned_int;
            };
      }
      else if ((cell_type == 3) && (dim == 3))
         // boundary info
      {
         subcelldata.boundary_quads.push_back (CellData<2>());
         in >> subcelldata.boundary_quads.back().vertices[0]
         >> subcelldata.boundary_quads.back().vertices[1]
         >> subcelldata.boundary_quads.back().vertices[2]
         >> subcelldata.boundary_quads.back().vertices[3];
         
         // to make sure that the cast wont fail
         Assert(material_id<= std::numeric_limits<types::boundary_id>::max(),
                ExcIndexRange(material_id,0,std::numeric_limits<types::boundary_id>::max()));
         // we use only boundary_ids in the range from 0 to numbers::internal_face_boundary_id-1
         Assert(material_id < numbers::internal_face_boundary_id,
                ExcIndexRange(material_id,0,numbers::internal_face_boundary_id));
         
         subcelldata.boundary_quads.back().boundary_id
         = static_cast<types::boundary_id>(material_id);
         
         // transform from gmsh to
         // consecutive numbering
         for (unsigned int i=0; i<4; ++i)
            if (vertex_indices.find (subcelldata.boundary_quads.back().vertices[i]) !=
                vertex_indices.end())
               // vertex with this index exists
               subcelldata.boundary_quads.back().vertices[i]
               = vertex_indices[subcelldata.boundary_quads.back().vertices[i]];
            else
            {
               // no such vertex index
               Assert (false,
                       ExcInvalidVertexIndex(cell,
                                             subcelldata.boundary_quads.back().vertices[i]));
               subcelldata.boundary_quads.back().vertices[i] =
               numbers::invalid_unsigned_int;
            };
         
      }
      else if (cell_type == 15)
      {
         // Ignore vertices
         // but read the
         // number of nodes
         // given
         switch (gmsh_file_format)
         {
            case 1:
            {
               for (unsigned int i=0; i<nod_num; ++i)
                  in >> dummy;
               break;
            }
            case 2:
            {
               in >> dummy;
               break;
            }
         }
      }
      else
         // cannot read this, so throw
         // an exception. treat
         // triangles and tetrahedra
         // specially since this
         // deserves a more explicit
         // error message
      {
         AssertThrow (cell_type != 2,
                      ExcMessage("Found triangles while reading a file "
                                 "in gmsh format. deal.II does not "
                                 "support triangles"));
         AssertThrow (cell_type != 11,
                      ExcMessage("Found tetrahedra while reading a file "
                                 "in gmsh format. deal.II does not "
                                 "support tetrahedra"));
         
         AssertThrow (false, ExcGmshUnsupportedGeometry(cell_type));
      }
   };
   
   // Assert we reached the end of the block
   in >> line;
   static const std::string end_elements_marker[] = {"$ENDELM", "$EndElements" };
   AssertThrow (line==end_elements_marker[gmsh_file_format-1],
                ExcInvalidGMSHInput(line));
   
   // check that no forbidden arrays are used
   Assert (subcelldata.check_consistency(dim), ExcInternalError());
   
   AssertThrow (in, ExcIO());
   
   // check that we actually read some
   // cells.
   AssertThrow(cells.size() > 0, ExcGmshNoCellInformation());
   
   // do some clean-up on
   // vertices...
   GridTools::delete_unused_vertices (vertices, cells, subcelldata);
   // ... and cells
   if (dim==spacedim)
      GridReordering<dim,spacedim>::invert_all_cells_of_negative_grid (vertices, cells);
   GridReordering<dim,spacedim>::reorder_cells (cells);
   tria->create_triangulation_compatibility (vertices, cells, subcelldata);
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void initialize_euler_vector (DoFHandler<dim> &dh,
                              GridData        &grid_data,
                              Vector<double>  &euler)
{
   const unsigned int   dofs_per_cell = dh.get_fe().dofs_per_cell;
   std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

   unsigned int c = 0;
   for (typename DoFHandler<dim>::active_cell_iterator
        cell = dh.begin_active(),
        endc = dh.end();
        cell!=endc; ++cell, ++c)
   {
      cell->get_dof_indices (local_dof_indices);
      
      for(unsigned int i=0; i<dofs_per_cell; ++i)
      {
         euler(local_dof_indices[i]) = 0.0;
      }
   }
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
int main()
{
   GridData grid_data;
   
   // read a gmsh file which contains isoparametric elements
   Triangulation<dim> triangulation;
   std::string filename = "./BL1_laminarJoukowskAirfoilGrids/Joukowski_Laminar_quad_ref0_Q4.msh";
   std::ifstream grid_file(filename.c_str());
   read_msh (grid_file, &triangulation, grid_data);

   std::cout << "Number of cells    = " << triangulation.n_cells() << std::endl;
   std::cout << "Number of vertices = " << triangulation.n_vertices() << std::endl;
   std::cout << "Cells in grid data = " << grid_data.v_in_cell.size() << std::endl;
   
   // write the grid to file. This writes a Q1 mesh.
   std::ofstream out ("grid_q1.vtk");
   GridOut grid_out;
   grid_out.write_vtk (triangulation, out);
   std::cout << "Q1 grid written into grid_q1.vtk\n";
   
   const FE_Q<dim> fe(1);
   const FESystem<dim> fesystem(fe, dim);
   DoFHandler<dim> dh(triangulation);
   dh.distribute_dofs(fesystem);
   const ComponentMask mask(dim, true);
   Vector<double> euler(dh.n_dofs());
   // Fills the euler vector with information from the Triangulation
   //VectorTools::get_position_vector(dhq, eulerq, mask);
   initialize_euler_vector (dh, grid_data, euler);
   MappingFEField<dim> map(dh, euler, mask);
}