# This file is part of psf_service_backend.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import tempfile
import unittest

import lsst.daf.butler
import lsst.utils.tests
from lsst.psf.service.backend import PsfServiceBackend, projection_finders


class TestPsfServiceBackend(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            # Note: we are taking advantage of the existing
            # `testdata_image_cutouts` here, as it contains the necessary
            # data to extract a PSF from and test the backend.
            cls.data_dir = lsst.utils.getPackageDir("testdata_image_cutouts")
        except LookupError:
            raise unittest.SkipTest(
                "PSFs must be extracted from the images stored in `testdata_image_cutouts` "
                "which is not set up."
            )

    def setUp(self):
        # Set up the butler using the test repository and collection.
        repo = os.path.join(self.data_dir, "repo")
        self.collection = "2.2i/runs/test-med-1/w_2022_03/DM-33223/20220118T193330Z"
        self.butler = lsst.daf.butler.Butler(repo, collections=self.collection)

        # Example dataset type known to contain a PSF.
        self.datasetType = "deepCoadd_calexp"

        # Example dataId within the test data repository.
        self.dataId = {"patch": 24, "tract": 3828, "band": "r", "skymap": "DC2"}

        # RA/Dec within the image; we choose coordinates inside the dataset
        # footprint.
        self.ra = 56.6400770
        self.dec = -36.4492250

        # For handling projections.
        self.projectionFinder = projection_finders.ReadComponents()

    def test_extract_psf(self):
        """Test PSF retrieval from a known dataset at a given RA/Dec."""
        ref = self.butler.registry.findDataset(self.datasetType, dataId=self.dataId)
        if ref is None:
            self.skipTest(f"No {self.datasetType} dataset found for dataId={self.dataId}.")

        with tempfile.TemporaryDirectory() as tempdir:
            psfBackend = PsfServiceBackend(self.butler, self.projectionFinder, tempdir)
            psfExtraction = psfBackend.extract_ref(self.ra, self.dec, ref)

            # Check that we got a valid PSF image.
            self.assertIsNotNone(psfExtraction.psf_image, "No PSF image returned.")
            psfImArray = psfExtraction.psf_image.array
            self.assertGreater(psfImArray.size, 0, "PSF image is empty.")
            self.assertNotEqual(psfImArray.sum(), 0.0, "PSF image appears to contain only zeros.")

            # Write out the result and confirm the file is produced.
            resultUri = psfBackend.write_fits(psfExtraction)
            self.assertTrue(os.path.exists(resultUri.ospath), "PSF FITS file was not created.")


if __name__ == "__main__":
    unittest.main()
