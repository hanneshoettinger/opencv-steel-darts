# `opencv-steel-darts` served with `django`

## Goals and ideas:

We would like to build a UI where a user is presented with
both the image capture of the game and the appropriate scores
as well as an option to run games and correct scores determined
by the image capture system.  
In the end, we hope to have a fully automated system to run
darts games

## Requirements:

* modified version of opencv-steel-darts
* mariaDB (open source DB), required to capture data
* django

## Current dependencies and set up details:

* Python version 3.5.x 
* Here's a list of the dependencies that are contained in the
`requirements.txt` file:

    ```
    astroid==1.6.1
    autopep8==1.3.4
    cycler==0.10.0
    Django==2.0.3
    django-mysql==2.2.0
    isort==4.3.4
    lazy-object-proxy==1.3.1
    matplotlib==2.1.2
    mccabe==0.6.1
    mysqlclient==1.3.12
    numpy==1.14.1
    pep8==1.7.1
    pycodestyle==2.3.1
    pylint==1.8.2
    pyparsing==2.2.0
    python-dateutil==2.6.1
    pytz==2018.3
    rope==0.10.7
    six==1.11.0
    wrapt==1.10.11
    ```
* mariaDB should be installed
  * for development a DB should be set up like in the `/orca_darts/settings.py` file:
    ```python
        DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': 'darts_cam',
            'USER': 'darts_admin',
            'PASSWORD': 'R04X7qpA1dAoZ0Q',
            'HOST': 'localhost',
            'PORT': '3306'
        }
    }
    ```
  * These are the commands used to initialize the DB and the admin user:
    ```mysql
    CREATE DATABASE darts_cam CHARACTER SET UTF8;
    CREATE USER darts_admin@localhost IDENTIFIED BY 'R04X7qpA1dAoZ0Q';
    GRANT ALL PRIVILEGES ON darts_cam.* TO darts_admin@localhost;
    FLUSH PRIVILEGES;
    ```
  * verify that everything went well ;-)
    ```mysql
    SHOW GRANTS FOR 'darts_admin'@'localhost';
    +--------------------------------------------------------------------------------------------------------------------+
    | Grants for darts_admin@localhost                                                                                   |
    +--------------------------------------------------------------------------------------------------------------------+
    | GRANT USAGE ON *.* TO 'darts_admin'@'localhost' IDENTIFIED BY PASSWORD '*CF4D3F9088A66610067706369E80E3870EA89F23' |
    | GRANT ALL PRIVILEGES ON `darts_cam`.* TO 'darts_admin'@'localhost'                                                 |
    +--------------------------------------------------------------------------------------------------------------------+
    2 rows in set (0.00 sec)
    
    mysql> SHOW CREATE USER 'darts_admin'@'localhost';
    +----------------------------------------------------------------------------------------------------------+
    | CREATE USER for darts_admin@localhost                                                                    |
    +----------------------------------------------------------------------------------------------------------+
    | CREATE USER 'darts_admin'@'localhost' IDENTIFIED BY PASSWORD '*CF4D3F9088A66610067706369E80E3870EA89F23' |
    +----------------------------------------------------------------------------------------------------------+
    1 row in set (0.00 sec)
    ```
* use `python manage.py createsuperuser` and follow the instructions to add a
`superuser` to administer the django instance from the web-ui (create/modify users)

## Starting and running the darts ui:

