# participant_metadata.csv

Sex and age (years) of the 50 autistic (`asd_N`) and 50 typically developing
(`td_N`) children, transcribed from the tables "Metadata of Typical
Participants" and "Metadata of Participants with ASD Information" in
`readme.docx` of the Dryad deposit (doi:10.5061/dryad.s7h44j150, CC0).

* `Case N` in each table is taken to be the child in folder `N` of the
  corresponding group (`Typical/N`, `Autism/children with ASD/N`), which is how
  `asd_N` and `td_N` are numbered in `processed/`. The deposit does not state
  this mapping explicitly.
* The ages in these tables span 5-15 years (autistic) and 2-14 years (typically
  developing), whereas the article describing the collection (Al-Jubouri et al.,
  2021, J. Phys.: Conf. Ser. 1818, 012201) gives 4-12 and 6-11 years.
* The tables also give a "Length (m)" column. Its values are not usable as
  height (they appear to be in feet for the typically developing children and
  in metres for the autistic children, and some are implausible for the stated
  age), so it is not included.
