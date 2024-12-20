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

from __future__ import annotations

__all__ = ("PsfServiceBackend", "PsfExtraction")

import dataclasses
from uuid import UUID, uuid4
from collections.abc import Sequence

from lsst.afw.image import ImageD
from lsst.daf.base import PropertyList
from lsst.daf.butler import Butler, DataId, DatasetRef
from lsst.resources import ResourcePath, ResourcePathExpression
from .projection_finders import ProjectionFinder
import lsst.geom as geom


@dataclasses.dataclass
class PsfExtraction:
    """A struct that stores the extracted PSF model at a point."""

    psf_image: ImageD
    """The PSF image itself."""

    metadata: PropertyList
    """Additional FITS metadata about the PSF extraction process."""

    origin_ref: DatasetRef
    """Fully-resolved reference to the dataset the PSF is from."""

    def write_fits(self, path: str) -> None:
        """Write the PSF image to a FITS file.

        Parameters
        ----------
        path : `str`
            Local path to the file.
        """
        self.psf_image.writeFits(fileName=path, metadata=self.metadata)


class PsfServiceBackend:
    """High-level interface to the PSF service backend.

    This backend can retrieve the PSF model from various LSST image datasets,
    for example:
    - `calexp` (Per-detector processed exposure),
    - `deepCoadd_calexp` (calibrated coadd exposures),
    - `deepDiff_differenceExp` (difference image produced in AP pipeline).

    Any dataset type that has a `getPsf()` method should be supported. Refer to
    the LSST Data Management code on GitHub (https://github.com/lsst) for more
    information on dataset types.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler that retrieves images of various types from the LSST Science
        Pipelines data repository.
    projection_finder : `ProjectionFinder`
        Object used to obtain WCS for butler datasets, allowing RA/Dec to pixel
        conversions.
    output_root : `lsst.resources.ResourcePathExpression`
        Root of output file URIs. The final PSF FITS file will be placed here.
    temporary_root : `lsst.resources.ResourcePathExpression`, optional
        Local filesystem root for writing temporary files before transferring
        to `output_root`.
    """

    def __init__(
        self,
        butler: Butler,
        projection_finder: ProjectionFinder,
        output_root: ResourcePathExpression,
        temporary_root: ResourcePathExpression | None = None,
    ):
        self.butler = butler
        self.projection_finder = projection_finder
        self.output_root = ResourcePath(output_root, forceAbsolute=True, forceDirectory=True)
        self.temporary_root = (
            ResourcePath(temporary_root, forceDirectory=True) if temporary_root is not None else None
        )

    butler: Butler
    projection_finder: ProjectionFinder
    output_root: ResourcePath
    temporary_root: ResourcePath | None

    def process_ref(self, ra: float, dec: float, ref: DatasetRef) -> ResourcePath:
        """Retrieve and write a PSF image from a fully-resolved `DatasetRef`.

        Parameters
        ----------
        ra, dec : `float`
            Right Ascension and Declination of the point (in degrees) where the
            PSF should be evaluated.
        ref : `lsst.daf.butler.DatasetRef`
            Fully-resolved reference to a dataset (e.g., `calexp`,
            `deepCoadd_calexp`, `goodSeeingDiff_differenceExp`).

        Returns
        -------
        uri : `lsst.resources.ResourcePath`
            Full path to the extracted PSF image file.
        """
        psf_result = self.extract_ref(ra, dec, ref)
        return self.write_fits(psf_result)

    def process_uuid(
        self,
        ra: float,
        dec: float,
        uuid: UUID,
        *,
        component: str | None = None,
    ) -> ResourcePath:
        """Retrieve and write a PSF image from a dataset identified by its
        UUID.

        Parameters
        ----------
        ra, dec : `float`
            RA/Dec of the point where the PSF should be evaluated (in degrees).
        uuid : `uuid.UUID`
            Unique ID of the dataset (e.g., a `calexp`).
        component : `str`, optional
            If not None, read this component instead of the composite dataset.

        Returns
        -------
        uri : `lsst.resources.ResourcePath`
            Full path to the extracted PSF image file.
        """
        psf_result = self.extract_uuid(ra, dec, uuid, component=component)
        return self.write_fits(psf_result)

    def process_search(
        self,
        ra: float,
        dec: float,
        dataset_type_name: str,
        data_id: DataId,
        collections: Sequence[str],
    ) -> ResourcePath:
        """Retrieve and write a PSF image from a dataset identified by
        (dataset type, data ID, collection).

        Parameters
        ----------
        ra, dec : `float`
            RA/Dec of the point where the PSF should be evaluated (in degrees).
        dataset_type_name : `str`
            Name of the butler dataset (e.g. "calexp", "deepCoadd_calexp",
            "goodSeeingDiff_differenceExp").
        data_id : `dict` or `lsst.daf.butler.DataCoordinate`
            Data ID used to find the dataset (e.g. {"visit": 12345,
            "detector": 42}).
        collections : `collections.abc.Sequence[str]`
            Collections to search for the dataset.

        Returns
        -------
        uri : `lsst.resources.ResourcePath`
            Full path to the extracted PSF image file.
        """
        psf_result = self.extract_search(ra, dec, dataset_type_name, data_id, collections)
        return self.write_fits(psf_result)

    def extract_ref(self, ra: float, dec: float, ref: DatasetRef) -> PsfExtraction:
        """Extract a PSF image from a fully-resolved `DatasetRef`.

        Parameters
        ----------
        ra, dec : `float`
            RA/Dec of the point where the PSF should be evaluated (in degrees).
        ref : `lsst.daf.butler.DatasetRef`
            Fully-resolved dataset reference.

        Returns
        -------
        psf_extraction : `PsfExtraction`
            Struct containing the PSF image, metadata and the origin reference.

        Raises
        ------
        ValueError
            If `ref.id` is not resolved or if the dataset does not contain a
            PSF.
        """
        if ref.id is None:
            raise ValueError(f"A resolved DatasetRef with a valid ID is required; got {ref}.")

        # Obtain WCS from the dataset and convert RA/Dec to pixels.
        wcs, _ = self.projection_finder(ref, self.butler)
        point_sky = geom.SpherePoint(geom.Angle(ra, geom.degrees), geom.Angle(dec, geom.degrees))
        point_pixel = wcs.skyToPixel(point_sky)

        # Get the image from the butler and extract the PSF model.
        image = self.butler.get(ref)

        if not hasattr(image, "getPsf"):
            raise ValueError(
                f"The dataset {ref.datasetType.name} with ID {ref.id} does not have a `getPSF()` method"
            )

        psf = image.getPsf()
        if psf is None:
            raise ValueError(f"No PSF found in dataset {ref.datasetType.name} with ID {ref.id}.")

        # Compute the PSF kernel image at the given point using the PSF model.
        psf_image = psf.computeKernelImage(point_pixel)

        # Create FITS metadata.
        metadata = PropertyList()
        metadata.set("BTLRUUID", ref.id.hex, "Butler dataset UUID from which this PSF was extracted.")
        metadata.set(
            "BTLRNAME", ref.datasetType.name, "Butler dataset type name from which this PSF was extracted."
        )
        metadata.set("PSFRA", ra, "RA of the PSF evaluation point (deg).")
        metadata.set("PSFDEC", dec, "Dec of the PSF evaluation point (deg).")

        for n, (k, v) in enumerate(ref.dataId.required.items()):
            metadata.set(f"BTLRK{n:03}", k, f"Name of dimension {n} in the data ID.")
            metadata.set(f"BTLRV{n:03}", v, f"Value of dimension {n} in the data ID.")

        return PsfExtraction(
            psf_image=psf_image,
            metadata=metadata,
            origin_ref=ref,
        )

    def extract_uuid(
        self, ra: float, dec: float, uuid: UUID, *, component: str | None = None
    ) -> PsfExtraction:
        """Extract a PSF image from a dataset identified by its UUID.

        Parameters
        ----------
        ra, dec : `float`
            RA/Dec (deg) of the PSF evaluation point.
        uuid : `UUID`
            Unique dataset identifier.
        component : `str`, optional
            If not None, read this component of the dataset.

        Returns
        -------
        psf_extraction : `PsfExtraction`
            Struct containing the PSF image, metadata and the origin reference.

        Raises
        ------
        LookupError
            If no dataset is found with the given UUID.
        """
        ref = self.butler.get_dataset(uuid)
        if ref is None:
            raise LookupError(f"No dataset found with UUID {uuid}.")
        if component is not None:
            ref = ref.makeComponentRef(component)

        return self.extract_ref(ra, dec, ref)

    def extract_search(
        self, ra: float, dec: float, dataset_type_name: str, data_id: DataId, collections: Sequence[str]
    ) -> PsfExtraction:
        """Extract a PSF image from a dataset identified by (dataset type,
        data ID, collection).

        Parameters
        ----------
        ra, dec : `float`
            RA/Dec (deg) of the PSF evaluation point.
        dataset_type_name : `str`
            Dataset type name of the image (e.g. "calexp", "deepCoadd_calexp",
            "goodSeeingDiff_differenceExp").
        data_id : `lsst.daf.butler.DataId`
            Data ID mapping used to locate the dataset.
        collections : `collections.abc.Sequence[str]`
            Collections to search for the dataset.

        Returns
        -------
        psf_extraction : `PsfExtraction`
            Struct containing the PSF image, metadata and the origin reference.

        Raises
        ------
        LookupError
            If no dataset is found with the given parameters.
        """
        ref = self.butler.find_dataset(dataset_type_name, data_id, collections=collections)
        if ref is None:
            raise LookupError(
                f"No {dataset_type_name} dataset found with data ID {data_id} in {collections}."
            )
        return self.extract_ref(ra, dec, ref)

    def write_fits(self, psf_result: PsfExtraction) -> ResourcePath:
        """Write a `PsfExtraction` to a FITS file in `output_root`.

        Parameters
        ----------
        psf_result : `PsfExtraction`
            The PSF extraction result to write.

        Returns
        -------
        uri : `lsst.resources.ResourcePath`
            Full path to the extracted PSF file.
        """
        output_uuid = uuid4()
        remote_uri = self.output_root.join(output_uuid.hex + ".fits")
        with ResourcePath.temporary_uri(prefix=self.temporary_root, suffix=".fits") as tmp_uri:
            tmp_uri.parent().mkdir()
            psf_result.write_fits(tmp_uri.ospath)
            remote_uri.transfer_from(tmp_uri, transfer="copy")
        return remote_uri
