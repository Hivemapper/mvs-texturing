/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef TEX_TEXTURING_HEADER
#define TEX_TEXTURING_HEADER

#include <vector>

#include "mve/mesh.h"
#include "mve/mesh_info.h"
#include <mvs_tex_mask/euclidean_view_mask.h>

#include "defines.h"
#include "obj_model.h"
#include "settings.h"
#include "sparse_table.h"
#include "texture_atlas.h"
#include "texture_patch.h"
#include "texture_view.h"
#include "uni_graph.h"

#include "seam_leveling.h"

TEX_NAMESPACE_BEGIN

using TextureViews = std::vector<TextureView>;
using TexturePatches = std::vector<TexturePatch::Ptr>;
using ConstTexturePatches = std::vector<TexturePatch::ConstPtr>;
using TextureAtlases = std::vector<TextureAtlas::Ptr>;
using Model = ObjModel;
using Graph = UniGraph ;
using DataCosts = SparseTable<std::uint32_t, std::uint16_t, float>;
using VertexProjectionInfos = std::vector<std::vector<VertexProjectionInfo>>;
using FaceProjectionInfos = std::vector<std::vector<FaceProjectionInfo>>;

/**
 * prepares the mesh for texturing
 *  -removes duplicated faces
 *  -ensures normals (face and vertex)
 */
void prepare_mesh(mve::MeshInfo* mesh_info, mve::TriangleMesh::Ptr mesh);

/**
 * Generates TextureViews from the in_scene.
 */
void generate_texture_views(
    std::string const& in_scene,
    TextureViews* texture_views,
    std::string const& tmp_dir);

/**
 * Builds up the meshes face adjacency graph using the vertex_infos
 */
void build_adjacency_graph(
    mve::TriangleMesh::ConstPtr mesh,
    mve::MeshInfo const& mesh_info,
    UniGraph* graph);

/**
 * Calculates the data costs for each face and texture view combination,
 * if the face is visible within the texture view.
 */
void calculate_data_costs(
    mve::TriangleMesh::ConstPtr mesh,
    TextureViews* texture_views,
    Settings const& settings,
    DataCosts* data_costs,
    std::shared_ptr<MvsTexturing::EuclideanViewMask> ev_mask,
    float* hidden_face_proportion = NULL);

void postprocess_face_infos(
    Settings const& settings,
    FaceProjectionInfos* projected_face_infos,
    DataCosts* data_costs);

/**
 * Runs the view selection procedure and saves the labeling in the graph
 */
void view_selection(
    DataCosts const& data_costs,
    UniGraph* graph,
    Settings const& settings);

/**
 * Generates texture patches using the graph to determine adjacent faces with
 * the same label.
 */
void generate_texture_patches(
    UniGraph const& graph,
    mve::TriangleMesh::ConstPtr mesh,
    mve::MeshInfo const& mesh_info,
    TextureViews* texture_views,
    Settings const& settings,
    VertexProjectionInfos* vertex_projection_infos,
    TexturePatches* texture_patches);

/**
 * Runs the seam leveling procedure proposed by Ivanov and Lempitsky
 * [<A
 * HREF="https://www.google.de/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&sqi=2&ved=0CC8QFjAA&url=http%3A%2F%2Fwww.robots.ox.ac.uk%2F~vilem%2FSeamlessMosaicing.pdf&ei=_ZbvUvSZIaPa4ASi7IGAAg&usg=AFQjCNGd4x5HnMMR68Sn2V5dPgmqJWErCA&sig2=4j47bXgovw-uks9LBGl_sA">Seamless
 * mosaicing of image-based texture maps</A>]
 */
void global_seam_leveling(
    UniGraph const& graph,
    mve::TriangleMesh::ConstPtr mesh,
    mve::MeshInfo const& mesh_info,
    VertexProjectionInfos const& vertex_projection_infos,
    TexturePatches* texture_patches);

void local_seam_leveling(
    UniGraph const& graph,
    mve::TriangleMesh::ConstPtr mesh,
    VertexProjectionInfos const& vertex_projection_infos,
    TexturePatches* texture_patches);

void local_seam_leveling_n(
    UniGraph const& graph,
    mve::TriangleMesh::ConstPtr mesh,
    VertexProjectionInfos const& vertex_projection_infos,
    TexturePatches* texture_patches,
    int num_texture_channels,
    TexturePatches* texture_object_class_patches = nullptr);

void generate_texture_atlases(
    TexturePatches * orig_texture_patches,
    Settings const & settings,
    TextureAtlases * texture_atlases,
    const std::vector<math::Vec3f>& vertices,
    const std::vector<uint>& mesh_faces);

void generate_capped_texture_atlas(
    TexturePatches * orig_texture_patches,
    Settings const & settings,
    TextureAtlases * texture_atlases,
    uint max_atlas_dim,
    const std::vector<math::Vec3f>& vertices,
    const std::vector<uint>& faces);

/**
 * Builds up an model for the mesh by constructing materials and
 * texture atlases form the texture_patches
 */
void build_model(
    mve::TriangleMesh::ConstPtr mesh,
    TextureAtlases const& texture_atlas,
    Model* model);

TEX_NAMESPACE_END

#endif /* TEX_TEXTURING_HEADER */
