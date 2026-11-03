# All 60 submission candidates: 3 poses per ligand from 3 independent sources.
#
#   open PyMOL in the folder holding A_*.pdb / E_*.pdb / G_*.pdb, then:  @view_candidates.pml
#
#   G_ (yellow)  GalaxyDock2-HEME, picked by docking score + crystal-motif match
#                -> 12/20 land inside the crystal Fe distance IQR (2.29-2.60 A)
#   A_ (green)   Boltz-2 consensus: medoid of the largest 2 A cluster
#   E_ (magenta) Boltz-2 motif: best agreement with the 14-residue crystal motif
#
# The three poses for one ligand come from different predictions and therefore
# different rigid-body frames, so everything is superposed on the protein first
# - without that they would appear tens of angstroms apart for no real reason.
#
# Reference values from 96 active-site CYP3A4 crystal complexes:
#   Fe-ligand distance  median 2.44 A  (IQR 2.29-2.60)
#   contacted residues  median 16
#   closest atom is N in 88% of them

python
from pymol import cmd
import os, glob
files = sorted(glob.glob("*.pdb"))
if not files:
    print("NO PDBs FOUND - start PyMOL in the directory containing A_*/E_*/G_*.pdb")
for f in files:
    cmd.load(f, os.path.splitext(os.path.basename(f))[0])
names = cmd.get_object_list()
ref = names[0]
for n in names[1:]:
    try:
        cmd.align(f"polymer and name CA and {n}", f"polymer and name CA and {ref}", cycles=5)
    except Exception as e:
        print("align failed:", n, e)
print(f"loaded and aligned {len(names)} candidate poses")
python end

hide everything
bg_color white

# one faint receptor copy for context
python
from pymol import cmd
ref = sorted(cmd.get_object_list())[0]
cmd.show("cartoon", f"{ref} and polymer")
cmd.color("grey80", f"{ref} and polymer")
cmd.set("cartoon_transparency", 0.8, ref)
python end

# heme + iron
show sticks, resn HEM
color firebrick, resn HEM and elem C
show spheres, resn HEM and elem FE
set sphere_scale, 0.35, resn HEM and elem FE
color orange, elem FE

# ligands, coloured by source
show sticks, resn LIG
color yellow,  G_* and resn LIG and elem C
color green,   A_* and resn LIG and elem C
color magenta, E_* and resn LIG and elem C
util.cnc("resn LIG and not elem C")
set stick_radius, 0.18, resn LIG
set stick_radius, 0.12, resn HEM

# the 14-residue consensus crystal motif
python
from pymol import cmd
motif = [119, 309, 212, 369, 305, 301, 213, 105, 108, 370, 215, 241, 120, 304]
ref = sorted(cmd.get_object_list())[0]
cmd.select("crystal_motif", f"{ref} and polymer and ({' or '.join('resi %d' % r for r in motif)})")
cmd.show("lines", "crystal_motif")
cmd.color("marine", "crystal_motif and elem C")
cmd.deselect()
python end

group GalaxyDock, G_*
group Boltz_consensus, A_*
group Boltz_motif, E_*

orient resn HEM
zoom resn HEM or resn LIG, 4
set valence, 0
set cartoon_side_chain_helper, 1

print "---------------------------------------------------------------"
print " yellow = GalaxyDock   green = Boltz consensus   magenta = Boltz motif"
print " orange sphere = heme Fe   blue lines = crystal contact motif"
print ""
print " one ligand at a time (all three sources):"
print "   disable *; enable *OCNT-2395436*; zoom resn LIG and enabled, 4"
print ""
print " one source only:   disable Boltz_motif"
print " Fe-N distances:    dist d, enabled and resn LIG and elem N, elem FE, 3.2"
print " crystal reference: Fe 2.44 A (IQR 2.29-2.60), 16 contacts, N closest in 88%"
print "---------------------------------------------------------------"
