# Day 7

## Convert ERDs into Relations

### Associative entities

break down many to many entities into associative entities

1. create a relation for each entity

2. In the relation for the associative entity add primary keys of the other two entities as its primary keys

For associative entities with their own IDs add the foreign keys to the relation

### Unary Relationship

Add a foreign key that refers to the primary key

many to many unary gets an associative entity

two ids original PK and the other instances PK

### Ternary Relationship

Add an associative entity and include all keys as pk and foreign keys in that entity
