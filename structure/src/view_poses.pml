# View the F_consensus and B_physics final poses for all 20 ligands.
#
#   cd ~/Projects/cyp-challenge/tmp/instance_20261103/pml
#   pymol ../../../structure/src/view_poses.pml
#
# Each PDB is a separate Boltz prediction in its OWN rigid-body frame, so they
# must be superposed before anything can be compared by eye - without the
# alignment below the ligands appear scattered tens of angstroms apart. The
# receptor itself is nearly identical between predictions (1.4 A CA RMSD), so
# a protein superposition is well defined.
#
# F = F_consensus (green)  : rank fusion of crystal motif + xTB dE + cluster population
# B = B_physics   (cyan)   : lowest xTB interaction energy
# Where green and cyan coincide, the two criteria agree (12/20 ligands).

# ---------------------------------------------------------------- load
python
from pymol import cmd
import os, glob
files = sorted(glob.glob("*.pdb"))
if not files:
    print("NO PDBs FOUND - run pymol from the directory holding B_*.pdb / F_*.pdb")
for f in files:
    cmd.load(f, os.path.splitext(os.path.basename(f))[0])
names = cmd.get_object_list()
ref = names[0]
# superpose every prediction onto the first using the protein chain only
for n in names[1:]:
    try:
        cmd.align(f"polymer and name CA and {n}", f"polymer and name CA and {ref}",
                  object=None, cycles=5)
    except Exception as e:
        print("align failed for", n, e)
print(f"loaded and aligned {len(names)} poses")
python end

# ---------------------------------------------------------------- style
hide everything
bg_color white
set ray_opaque_background, 0

# receptor: one faint cartoon from the reference only, as all are near-identical
python
from pymol import cmd
ref = sorted(cmd.get_object_list())[0]
cmd.show("cartoon", f"{ref} and polymer")
cmd.color("grey80", f"{ref} and polymer")
cmd.set("cartoon_transparency", 0.75, ref)
python end

# heme from the reference, as sticks
show sticks, resn HEM
color firebrick, resn HEM and elem C
show spheres, resn HEM and elem FE
set sphere_scale, 0.35, resn HEM and elem FE
color orange, elem FE

# ligands: green = F_consensus, cyan = B_physics
show sticks, resn LIG
color green, F_* and resn LIG and elem C
color cyan,  B_* and resn LIG and elem C
util.cnc("resn LIG and not elem C")

# the 14-residue consensus crystal motif (>=50% of 96 known CYP3A4 complexes)
python
from pymol import cmd
motif = [119, 309, 212, 369, 305, 301, 213, 105, 108, 370, 215, 241, 120, 304]
ref = sorted(cmd.get_object_list())[0]
sel = " or ".join([f"resi {r}" for r in motif])
cmd.select("crystal_motif", f"{ref} and polymer and ({sel})")
cmd.show("lines", "crystal_motif")
cmd.color("marine", "crystal_motif and elem C")
cmd.deselect()
python end

# the axial cysteine that anchors the heme
select cys442, resi 442 and not resn LIG
show sticks, cys442
color yellow, cys442 and elem S
deselect

# ---------------------------------------------------------------- view
orient resn HEM
zoom resn HEM or resn LIG, 4
set stick_radius, 0.12
set stick_radius, 0.18, resn LIG
set valence, 0
set cartoon_side_chain_helper, 1

# Grouping so individual ligands can be toggled quickly.
group F_consensus, F_*
group B_physics, B_*

print "-----------------------------------------------------------"
print " green = F_consensus   cyan = B_physics   blue lines = crystal motif"
print " orange sphere = heme Fe   yellow = Cys442 thiolate"
print ""
print " disable F_consensus      # show only B"
print " disable B_physics        # show only F"
print " enable  F_OCNT-2395436   # one ligand at a time"
print " dist d, resn LIG and elem N, elem FE, 3.0   # check Fe-N distances"
print "-----------------------------------------------------------"
