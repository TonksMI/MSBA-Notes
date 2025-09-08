# Day 2

## Attributes

### Required attribute vs optional

required attributes must have a value for every instance the which it is associated
we cannot store any data about an employee without

### Composite attribute vs simple attributes

Composite has meaningful components

name can be first_name last_name instead
address can be broken down into each portion of a line

composite attributes are useful and simple attributes are also useful. Depending on the scenario we might need to use either. If we were trying to get a State out of an address or city to do location data we need to split an address into its parts

Atomic attributes can't be broken down

### Composite Identifier

an Identifier that consists of a composite attribute

- Example combine (flight number + Date)

Simple Identifier that consists of a simple attribute

- can be table generated

### Single Valued vs multivalued

multi valued attributes

- attributes that can take more than one value

{column} is used to define multivalued columns

### Stored vs derived

columns that can be derived from a column using logic

date_employed -> years_employed

## Entity

### Strong vs weak entity

Strong entities can live independently of other types

- Shown in a rectangle in ERD

Weak entities existence depend on some other entity type

- weak entities are in a double lined rectangle

### Owner

when entity type owns another entity

Partial Identifier

Identifier of a weak Identifier

In combination of one entity's identifier + one partial entity from the child is unique

## Relationship

### Data Models

capture the nature and relationships amond data using a graphical system

contains entities attributes and relationships

### ERD

the graphical system/ showing the value

### Relationships

the links between entities shown by lines connecting the entities

Relationships are labeled with verb phrases

The symbols at the end of each line specify cardinalities

- one to one (straight line)

- one to many (starts with a normal line with a triangle at the end)

- many to many (start and end with a triangle)

