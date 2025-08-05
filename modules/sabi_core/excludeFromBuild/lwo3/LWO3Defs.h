#pragma once

// LWO3Defs: Contains definitions and utility functions for LWO3 file format
//
// This file typically includes:
// - Enum or constexpr definitions for various LWO3 chunk and form IDs
// - Utility functions for working with LWO3 data
// - Any other constants or types specific to the LWO3 format

// #include <mace_core/mace_core.h>

using mace::swap16;
using mace::swap32;
using mace::swap64;
using mace::swapDouble;
using mace::swapFloat;

namespace LWO
{
    // make IDs
    constexpr uint32_t make_id (char a, char b, char c, char d)
    {
        return (static_cast<uint32_t> (a) << 24) |
               (static_cast<uint32_t> (b) << 16) |
               (static_cast<uint32_t> (c) << 8) |
               static_cast<uint32_t> (d);
    }

    // Forms
    constexpr uint32_t FORM = make_id ('F', 'O', 'R', 'M'); // IFF container form
    constexpr uint32_t LWO3 = make_id ('L', 'W', 'O', '3'); // LightWave Object 2018 form
    constexpr uint32_t CLIP = make_id ('C', 'L', 'I', 'P'); // Image/sequence reference
    constexpr uint32_t STIL = make_id ('S', 'T', 'I', 'L'); // Still image
    constexpr uint32_t SURF = make_id ('S', 'U', 'R', 'F'); // Surface definition
    constexpr uint32_t PART = make_id ('P', 'A', 'R', 'T'); // Part definition
    constexpr uint32_t NODS = make_id ('N', 'O', 'D', 'S'); // Node definitions
    constexpr uint32_t NROT = make_id ('N', 'R', 'O', 'T'); // Node root
    constexpr uint32_t NNDS = make_id ('N', 'N', 'D', 'S'); // Node list
    constexpr uint32_t NTAG = make_id ('N', 'T', 'A', 'G'); // Node tag
    constexpr uint32_t NDTA = make_id ('N', 'D', 'T', 'A'); // Node data
    constexpr uint32_t IBGC = make_id ('I', 'B', 'G', 'C'); // Image background color
    constexpr uint32_t IOPC = make_id ('I', 'O', 'P', 'C'); // Image opacity
    constexpr uint32_t IMST = make_id ('I', 'M', 'S', 'T'); // Image state
    constexpr uint32_t IIMG = make_id ('I', 'I', 'M', 'G'); // Image image
    constexpr uint32_t IBMP = make_id ('I', 'B', 'M', 'P'); // Image bitmap
    constexpr uint32_t IUTD = make_id ('I', 'U', 'T', 'D'); // Image U translation
    constexpr uint32_t IVTD = make_id ('I', 'V', 'T', 'D'); // Image V translation
    constexpr uint32_t ISCL = make_id ('I', 'S', 'C', 'L'); // Image scale
    constexpr uint32_t IPOS = make_id ('I', 'P', 'O', 'S'); // Image position
    constexpr uint32_t IROT = make_id ('I', 'R', 'O', 'T'); // Image rotation
    constexpr uint32_t IFAL = make_id ('I', 'F', 'A', 'L'); // Image falloff
    constexpr uint32_t NCON = make_id ('N', 'C', 'O', 'N'); // Node connections
    constexpr uint32_t SSHA = make_id ('S', 'S', 'H', 'A'); // Surface shader
    constexpr uint32_t SSHD = make_id ('S', 'S', 'H', 'D'); // Surface shader data
    constexpr uint32_t ATTR = make_id ('A', 'T', 'T', 'R'); // Attributes
    constexpr uint32_t META = make_id ('M', 'E', 'T', 'A'); // Metadata
    constexpr uint32_t ADAT = make_id ('A', 'D', 'A', 'T'); // Attribute data
    constexpr uint32_t ENTR = make_id ('E', 'N', 'T', 'R'); // Entry
    constexpr uint32_t VALU = make_id ('V', 'A', 'L', 'U'); // Value
    constexpr uint32_t VPRM = make_id ('V', 'P', 'R', 'M'); // Vertex parameters
    constexpr uint32_t VPVL = make_id ('V', 'P', 'V', 'L'); // Vertex parameter value

    // normal map
    constexpr uint32_t IINX = make_id ('I', 'I', 'N', 'X');
    constexpr uint32_t IINY = make_id ('I', 'I', 'N', 'Y');
    constexpr uint32_t IINZ = make_id ('I', 'I', 'N', 'Z');

    // Geometry
    constexpr uint32_t TAGS = make_id ('T', 'A', 'G', 'S'); // Surface tags
    constexpr uint32_t LAYR = make_id ('L', 'A', 'Y', 'R'); // Layer
    constexpr uint32_t PNTS = make_id ('P', 'N', 'T', 'S'); // Points
    constexpr uint32_t BBOX = make_id ('B', 'B', 'O', 'X'); // Bounding box
    constexpr uint32_t VMPA = make_id ('V', 'M', 'P', 'A'); // Vertex map parameters
    constexpr uint32_t VMAP = make_id ('V', 'M', 'A', 'P'); // Vertex map
    constexpr uint32_t POLS = make_id ('P', 'O', 'L', 'S'); // Polygons
    constexpr uint32_t OTAG = make_id ('O', 'T', 'A', 'G'); // Object tag

    // Polygon types
    constexpr uint32_t FACE = make_id ('F', 'A', 'C', 'E'); // Face type polygons
    constexpr uint32_t CURV = make_id ('C', 'U', 'R', 'V'); // Curve type polygons
    constexpr uint32_t PTCH = make_id ('P', 'T', 'C', 'H'); // Patch type polygons
    constexpr uint32_t MBAL = make_id ('M', 'B', 'A', 'L'); // Metaball type polygons
    constexpr uint32_t BONE = make_id ('B', 'O', 'N', 'E'); // Bone type polygons

    // Vertex map types
    constexpr uint32_t PICK = make_id ('P', 'I', 'C', 'K'); // Selection set
    constexpr uint32_t WGHT = make_id ('W', 'G', 'H', 'T'); // Weight maps
    constexpr uint32_t MNVW = make_id ('M', 'N', 'V', 'W'); // Subpatch weight maps
    constexpr uint32_t TXUV = make_id ('T', 'X', 'U', 'V'); // UV texture coordinates
    constexpr uint32_t MORF = make_id ('M', 'O', 'R', 'F'); // Morph target maps
    constexpr uint32_t SPOT = make_id ('S', 'P', 'O', 'T'); // Spot maps
    constexpr uint32_t RGB = make_id ('R', 'G', 'B', ' ');  // RGB color maps
    constexpr uint32_t RGBA = make_id ('R', 'G', 'B', 'A'); // RGBA color maps

    // Surfaces
    constexpr uint32_t COLR = make_id ('C', 'O', 'L', 'R'); // Base color
    constexpr uint32_t LUMI = make_id ('L', 'U', 'M', 'I'); // Luminosity
    constexpr uint32_t DIFF = make_id ('D', 'I', 'F', 'F'); // Diffuse
    constexpr uint32_t SPEC = make_id ('S', 'P', 'E', 'C'); // Specularity
    constexpr uint32_t GLOS = make_id ('G', 'L', 'O', 'S'); // Glossiness
    constexpr uint32_t REFL = make_id ('R', 'E', 'F', 'L'); // Reflection
    constexpr uint32_t RFOP = make_id ('R', 'F', 'O', 'P'); // Reflection options
    constexpr uint32_t RIMG = make_id ('R', 'I', 'M', 'G'); // Reflection image
    constexpr uint32_t RSAN = make_id ('R', 'S', 'A', 'N'); // Reflection seam angle
    constexpr uint32_t TRAN = make_id ('T', 'R', 'A', 'N'); // Transparency
    constexpr uint32_t TROP = make_id ('T', 'R', 'O', 'P'); // Transparency options
    constexpr uint32_t TIMG = make_id ('T', 'I', 'M', 'G'); // Transparency image
    constexpr uint32_t RIND = make_id ('R', 'I', 'N', 'D'); // Refraction index
    constexpr uint32_t TRNL = make_id ('T', 'R', 'N', 'L'); // Translucency
    constexpr uint32_t BUMP = make_id ('B', 'U', 'M', 'P'); // Bump
    constexpr uint32_t SMAN = make_id ('S', 'M', 'A', 'N'); // Max smoothing angle
    constexpr uint32_t SIDE = make_id ('S', 'I', 'D', 'E'); // Sidedness
    constexpr uint32_t CLRH = make_id ('C', 'L', 'R', 'H'); // Color highlights
    constexpr uint32_t CLRF = make_id ('C', 'L', 'R', 'F'); // Color filter
    constexpr uint32_t ADTR = make_id ('A', 'D', 'T', 'R'); // Additive transparency
    constexpr uint32_t SHRP = make_id ('S', 'H', 'R', 'P'); // Diffuse sharpness
    constexpr uint32_t LSIZ = make_id ('L', 'S', 'I', 'Z'); // Line size
    constexpr uint32_t ALPH = make_id ('A', 'L', 'P', 'H'); // Alpha mode
    constexpr uint32_t AVAL = make_id ('A', 'V', 'A', 'L'); // Alpha value
    constexpr uint32_t GVAL = make_id ('G', 'V', 'A', 'L'); // Glow value
    constexpr uint32_t BLOK = make_id ('B', 'L', 'O', 'K'); // Render block
    constexpr uint32_t TMAP = make_id ('T', 'M', 'A', 'P'); // Texture map

    constexpr uint32_t PTAG = make_id ('P', 'T', 'A', 'G'); // Polygon tags
    constexpr uint32_t VMAD = make_id ('V', 'M', 'A', 'D'); // Discontinuous vertex mapping
    constexpr uint32_t FLAG = make_id ('F', 'L', 'A', 'G'); // Flags

    constexpr uint32_t VERS = make_id ('V', 'E', 'R', 'S'); // Version
    constexpr uint32_t NLOC = make_id ('N', 'L', 'O', 'C'); // Node location
    constexpr uint32_t NZOM = make_id ('N', 'Z', 'O', 'M'); // Node zoom
    constexpr uint32_t NSEL = make_id ('N', 'S', 'E', 'L'); // Node selection state
    constexpr uint32_t NCOL = make_id ('N', 'C', 'O', 'L'); // Node column
    constexpr uint32_t NFRS = make_id ('N', 'F', 'R', 'S'); // Node plugin name
    constexpr uint32_t NSTA = make_id ('N', 'S', 'T', 'A'); // Node state
    constexpr uint32_t NVER = make_id ('N', 'V', 'E', 'R'); // Node version
    constexpr uint32_t NSRV = make_id ('N', 'S', 'R', 'V'); // Node server
    constexpr uint32_t NRNM = make_id ('N', 'R', 'N', 'M'); // Node real name
    constexpr uint32_t NNME = make_id ('N', 'N', 'M', 'E'); // Node name
    constexpr uint32_t NCRD = make_id ('N', 'C', 'R', 'D'); // Node coordinates
    constexpr uint32_t NMOD = make_id ('N', 'M', 'O', 'D'); // Node mode
    constexpr uint32_t NPRW = make_id ('N', 'P', 'R', 'W'); // Node preview
    constexpr uint32_t NCOM = make_id ('N', 'C', 'O', 'M'); // Node comment
    constexpr uint32_t IPIX = make_id ('I', 'P', 'I', 'X'); // Image pixel filter
    constexpr uint32_t IMIP = make_id ('I', 'M', 'I', 'P'); // Image mipmap
    constexpr uint32_t IMOD = make_id ('I', 'M', 'O', 'D'); // Image edit mode
    constexpr uint32_t IINV = make_id ('I', 'I', 'N', 'V'); // Image invert
    constexpr uint32_t INCR = make_id ('I', 'N', 'C', 'R'); // Image normal correction
    constexpr uint32_t IMAP = make_id ('I', 'M', 'A', 'P'); // Image mapping
    constexpr uint32_t IAXS = make_id ('I', 'A', 'X', 'S'); // Image axis
    constexpr uint32_t IFOT = make_id ('I', 'F', 'O', 'T'); // Image falloff type
    constexpr uint32_t IWRL = make_id ('I', 'W', 'R', 'L'); // Image wrap layers
    constexpr uint32_t IREF = make_id ('I', 'R', 'E', 'F'); // Image reference
    constexpr uint32_t IUVI = make_id ('I', 'U', 'V', 'I'); // Image UV index
    constexpr uint32_t IUTI = make_id ('I', 'U', 'T', 'I'); // Image U tile
    constexpr uint32_t ITIM = make_id ('I', 'T', 'I', 'M'); // Image time
    constexpr uint32_t IUTL = make_id ('I', 'U', 'T', 'L'); // Image U tile loop
    constexpr uint32_t IVTL = make_id ('I', 'V', 'T', 'L'); // Image V tile loop
    constexpr uint32_t INME = make_id ('I', 'N', 'M', 'E'); // Input node name
    constexpr uint32_t IINM = make_id ('I', 'I', 'N', 'M'); // Input name
    constexpr uint32_t IINN = make_id ('I', 'I', 'N', 'N'); // Input node name
    constexpr uint32_t IONM = make_id ('I', 'O', 'N', 'M'); // Input output name
    constexpr uint32_t SSHN = make_id ('S', 'S', 'H', 'N'); // Surface shader name
    constexpr uint32_t NAME = make_id ('N', 'A', 'M', 'E'); // Name
    constexpr uint32_t ENUM = make_id ('E', 'N', 'U', 'M'); // Enumeration
    constexpr uint32_t TAG = make_id ('T', 'A', 'G', ' ');  // Tag
    constexpr uint32_t NPLA = make_id ('N', 'P', 'L', 'A'); // Node placement
    constexpr uint32_t AOVS = make_id ('A', 'O', 'V', 'S'); // Arbitrary Output Variables.
    constexpr uint32_t SATR = make_id ('S', 'A', 'T', 'R'); // Standard Material Attribute maybe??
    // Anonymous chunk (four spaces)
    constexpr uint32_t ANON = 0x20202020;

    // these are temporary ANON replacements, they will still be written is 4 spaces
    constexpr uint32_t SNAM = make_id ('S', 'N', 'A', 'M'); // Surface Name
    constexpr uint32_t CIDX = make_id ('C', 'I', 'D', 'X'); // Clip Index
    constexpr uint32_t SFNM = make_id ('S', 'F', 'N', 'M'); // Still File Name
    constexpr uint32_t VTYP = make_id ('V', 'T', 'Y', 'P'); // Value Type
    constexpr uint32_t VVAL = make_id ('V', 'V', 'A', 'L'); // Value Value
    constexpr uint32_t IVAL = make_id ('I', 'V', 'A', 'L'); // Image Value
    constexpr uint32_t VDAT = make_id ('V', 'D', 'A', 'T'); // Value Data

    // temp standin for PTAGs
    constexpr uint32_t PTAG_SURF = make_id ('P', 'T', 'G', 'S'); // PTAG for Surface
    constexpr uint32_t PTAG_COLR = make_id ('P', 'T', 'G', 'C'); // PTAG for Color

    constexpr uint32_t CLRS = make_id ('C', 'L', 'R', 'S');
    constexpr uint32_t CLRA = make_id ('C', 'L', 'R', 'A');
} // namespace LWO

inline std::string idToString (uint32_t id)
{
    std::array<char, 5> str;
    str[0] = static_cast<char> ((id >> 24) & 0xFF);
    str[1] = static_cast<char> ((id >> 16) & 0xFF);
    str[2] = static_cast<char> ((id >> 8) & 0xFF);
    str[3] = static_cast<char> (id & 0xFF);
    str[4] = '\0';
    return std::string (str.data());
}