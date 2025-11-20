# Copyright (C) 2025  Technische Universitaet Berlin
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA

"""
The conftest.py file serves as a means of providing fixtures for an entire directory.
Fixtures defined in a conftest.py can be used by any test in that package without needing to
import them (pytest will automatically discover them).
"""

import pytest
# from sqlalchemy import create_engine
# from create_db_schema import create_schema, create_db, drop_db
from xicommon.config import Config, Crosslinker, Enzyme, DigestionConfig


# @pytest.fixture()
# def db_info():
#     # returns the test database credentials
#     return {
#         "hostname": str(os.environ.get('XI2TEST_PSQL_HOST', 'localhost')),
#         "port": str(os.environ.get('XI2TEST_PSQL_PORT', '5432')),
#         "db": str(os.environ.get('XI2TEST_PSQL_DB', 'xisearch2_unittest')),
#         "dbuser": str(os.environ.get('XI2TEST_PSQL_USER', 'xisearch2_unittest')),
#         "dbpassword": str(os.environ.get('XI2TEST_PSQL_PASSWORD', 'xisearch2_unittest')),
#     }
#
#
# @pytest.fixture()
# def engine(db_info):
#     # A new SqlAlchemy connection to the test database
#     return create_engine(
#         f"postgresql://{db_info['dbuser']}"
#         f":{db_info['dbpassword']}@{db_info['hostname']}/{db_info['db']}"
#     )
#
#
# @pytest.fixture()
# def use_database(db_info):
#     # Create a tempary test Postgresql database
#     create_db(
#         db=db_info["db"], dbuser=db_info["dbuser"], dbpassword=db_info["dbpassword"]
#     )
#     create_schema(
#         db=db_info["db"], dbuser=db_info["dbuser"], dbpassword=db_info["dbpassword"]
#     )
#     yield
#     drop_db(
#         db=db_info["db"], dbuser=db_info["dbuser"], dbpassword=db_info["dbpassword"]
#     )


@pytest.fixture()
def search_config():
    # Basic Search config to use for tests
    return Config(
        reporting_requirements={'report_top_ranking_only': False},
        digestion=DigestionConfig(
            enzymes=[Enzyme.trypsin], missed_cleavages=0, min_peptide_length=3
        ),
        ms1_tol="5ppm",
        ms2_tol="15ppm",
        fragmentation={"add_precursor": False},
        top_n_alpha_scores=10,
        crosslinker=[Crosslinker.BS3],
    )
