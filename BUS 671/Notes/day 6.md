# Day 6 after first midterm

## Data Model

AN abstract representation of the structure and organization of data along with the rules for how that data is stored related and manipulated

1 describing data

2 constraining data

3 manipulating data

## Conceptual data model

ER Data model

conceptual model represents data from teh viewpoint of the org independent of any technology that will be used to implement the model

Conceptual data model is about understanding the organization getting the database requirements right

Conceptual schema

the output of the conceptual data modeling is called a conceptual schema

A sonceptual schema is a detailed technology independent specification

agile

Product manager

Scrum Master

Team Manager

Logical Data Model

A data model that is consistent and compatible with a specific type of database technology

logical schema

the representation of a database for a particular data management technology

conceptual model ehat data and what relationships are needed

logical model how does this look in the type of database we are using

physical model is how it is implemented in the database

### Objectives

## Concepts

### Relation

a named 2d table of data

relation - entity type

relation in relational database <> relationship in er model

#### Properties

six requirements

- it must have a unique name
- Every attribute value must be atomic
- every row must be unique
- columns in tables must have unique names
- the order of columns is irrelevant
- the order of the rows are irrelevant

notation

previously for text notation :

EMPLOYEE(EMPID,Name,Dept,Salary)

grpahical:

EMPLOYEE

(in a box)

[EmpId|Name|Dept|Salary]

#### Primary key

an attribute or a combination of attributes that uniquely identifies each row in a relation

Primary key helps to find the record we need from database

primary keys are underlined

Note

an entites identifier in an ER diagram may or may not be the same attributes that comprise the primary key for the relation and may be a combination

Composite key

primary key that consists of more than one attribute

underline identifiers

best practice to keep the ids at the start unless if the columns are logically grouped

##### Integrety Constraints

Integrity constraints

rules that facilitate maintaining the accuracy and integrity of data in the database

Entity integrity constraint

a rule that states no primary key attribute can be null

null value that indicates nothing in that 'cell'

Referential integrity constraint

foreign keys

#### Foreign key

a key in a relation as an attribute that serves as the primary key of another relation in the same database

the foreign key is emphasized by using a dashed underline

in relation data model associations between tables are defined. through the use of foreign keys

the link between the department and employee table is through DeptName

This implies that before we insert a new row int he employee table the department for that employee must already exist in the department table

referential integrity constraint

either each foreign key value must match a primary key value in another relation or the foreign key value must be null

if it is boptional to have a foreign key it can be null

## Composite attributes

split into multiple columns

ERD has it as a composite

relation has it as their own columns. keep only the parts of the attribute

## Multivalued attributes

split off into its own relation with the initial primary key and the split off attributes as the other primary key

weak entites become their own relation with a primary and foreign key for refering to the strong entity

many to many

associative entity still holds but there is more
