import mysql.connector
from mysql.connector import Error

def create_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

connection = create_connection("localhost", "root", "Tung3071999%")

def create_database(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as e:
        print(f"The error '{e}' occurred")
create_database_query = "CREATE DATABASE  if not exists  data_student"
create_database(connection, "CREATE DATABASE   if not exists  data_student")
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="Tung3071999%",
  database="sm_app"
)

mycursor = mydb.cursor(buffered=True)
mycursor.execute("CREATE TABLE if not exists Student (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), address VARCHAR(255), time VARCHAR(255))")

mycursor.execute("CREATE TABLE if not exists Checkin  (id INT AUTO_INCREMENT PRIMARY KEY,name VARCHAR(255), time TIME)")
mycursor.execute("SELECT * FROM Checkin JOIN Student USING(name)")

sql = "INSERT INTO Student(name, address) VALUES (%s,%s)"
val = ("John", "132")

mycursor.execute(sql,val)
mydb.commit()