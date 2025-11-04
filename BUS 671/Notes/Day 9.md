# Midterm

## Normalization 2

1st form

unique rows and each row is atomic

2nd

no partial dependency

3rd

no transitive dependencies

## Transitive dependency

occurs when some attributes depend on some other attribute that does not uniquely identify each row in that table

Candidate key is an attribute that can uniquely identify each row in a relation

the primary key is the candidate key that you choose to uniquely identify rows in a table

non primary attribtue is an attribute that is not part of any candidate key

exists when attributes depend on other non primary attirbutes

transitive dependencies create unnecessary redundancy

## Second normal form to 3rd normal form

Identify the trans dependencies

remove them by creating new tables

1. move dependent attributes to a new table

2. make the attribute they depend on the primary key of the new table

3. define foreign key from original table

