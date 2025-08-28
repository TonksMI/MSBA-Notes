# Day 1

## Managing data for analysis

### Data

#### Definition

Facts concerning objects and events that could be recorded and stored on computer

#### Examples

Facts concerning a student Name major DOB address phone number

Factes concerning car repair shop Photos of cards scanned images of forms

#### Atomic transactions

unless everything goes through and nothing goes through

### Types of data

#### structured data

data that can be stored in excel tables

numbers characters data

#### Unstructured data

Data that can not be stored

images sound videos emails tweets Facebook

## what is different between data and information

information

- data the has been processed in such a way that can increase the knowledge of the person using the data

- How to transform data

  - add context to the data
  - extract patterns

## Meta Data

- form of data that describes the properties of the main data

- Helps understanding what the data means

## Exponential growth

amount of data has been growing exponentially

## How to approach data

Use a database to collect logically related data

Health care db

- patient data
- appointment data

## Old fashion file processing system

Collection of related files stored in one place to store and process data

File processing systems

- collection of files and programs access and modify the files

- files is a collection of related records

### Disadvantages

- Programs can create read update and delete files
- duplication of data
- file-program dependence
- data sharing limitation
- data inconsistency

## Database approach

Central database is best

## Database Management system

- A software that enables systematic method of creating storing updating and retreiving data
- DBMS enables data to be shared among multiple users and applications rather than stored in files
- A DBMS also provides facilities for controlling data access enforcing data integrity managing concurrency control and restoring a database

## Data base design

How to link between each table to be able to interact with each other to get more information

## Advantages

- eliminate redundency
- program data independence
- data consistency
- engorcement of standards
- data quality
- improved access

## Costs and risks

needs new specialized personel

- Need to hire/train people to design and implement dbs and become dba's

- installation and management cost and complexity

- conversion and costs

- need for backup and recovery when damage occurs

## The tech behind dbms

- data modeling


### Data Modeling (tables)

entities a person place concept which the org wishes to maintain data

example of entity type:

- Person
- Place
- Object
- Event
- Concept

#### Attributes (columns)

Characteristics about that entity
If we can attribute a name to one word

##### format for professor:

Uppercase

Uppercase_Lowercase

undeline keys

### Entities

Entity type is a collection of entities that share common properties

Entity instance (record)

### Entity Identifier

An attribute that contains a unique value to distinguis a record(instance)

no two records can have the same ID

Underline the IDs